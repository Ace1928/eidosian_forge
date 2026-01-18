import logging
from collections import defaultdict
from pyomo.common.autoslots import AutoSlots
import pyomo.common.config as cfg
from pyomo.common import deprecated
from pyomo.common.collections import ComponentMap, ComponentSet, DefaultComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.expr.numvalue import ZeroConstant
import pyomo.core.expr as EXPR
from pyomo.core.base import TransformationFactory, Reference
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.disjunct import _DisjunctData
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import (
from pyomo.core.util import target_list
from pyomo.util.vars_from_expressions import get_vars_from_components
from weakref import ref as weakref_ref
@TransformationFactory.register('gdp.hull', doc='Relax disjunctive model by forming the hull reformulation.')
class Hull_Reformulation(GDP_to_MIP_Transformation):
    """Relax disjunctive model by forming the hull reformulation.

    Relaxes a disjunctive model into an algebraic model by forming the
    hull reformulation of each disjunction.

    This transformation accepts the following keyword arguments:

    Parameters
    ----------
    perspective_function : str
        The perspective function used for the disaggregated variables.
        Must be one of 'FurmanSawayaGrossmann' (default),
        'LeeGrossmann', or 'GrossmannLee'
    EPS : float
        The value to use for epsilon [default: 1e-4]
    targets : (block, disjunction, or list of those types)
        The targets to transform. This can be a block, disjunction, or a
        list of blocks and Disjunctions [default: the instance]

    The transformation will create a new Block with a unique
    name beginning "_pyomo_gdp_hull_reformulation". It will contain an
    indexed Block named "relaxedDisjuncts" that will hold the relaxed
    disjuncts.  This block is indexed by an integer indicating the order
    in which the disjuncts were relaxed. Each block has a dictionary
    "_constraintMap":

        'srcConstraints': ComponentMap(<transformed constraint>:
                                       <src constraint>),
        'transformedConstraints':
            ComponentMap(<src constraint container> :
                         <transformed constraint container>,
                         <src constraintData> : [<transformed constraintDatas>])

    All transformed Disjuncts will have a pointer to the block their transformed
    constraints are on, and all transformed Disjunctions will have a
    pointer to the corresponding OR or XOR constraint.

    The _pyomo_gdp_hull_reformulation block will have a ComponentMap
    "_disaggregationConstraintMap":
        <src var>:ComponentMap(<srcDisjunction>: <disaggregation constraint>)
    """
    CONFIG = cfg.ConfigDict('gdp.hull')
    CONFIG.declare('targets', cfg.ConfigValue(default=None, domain=target_list, description='target or list of targets that will be relaxed', doc='\n\n        This specifies the target or list of targets to relax as either a\n        component or a list of components. If None (default), the entire model\n        is transformed. Note that if the transformation is done out of place,\n        the list of targets should be attached to the model before it is cloned,\n        and the list will specify the targets on the cloned instance.'))
    CONFIG.declare('perspective function', cfg.ConfigValue(default='FurmanSawayaGrossmann', domain=cfg.In(['FurmanSawayaGrossmann', 'LeeGrossmann', 'GrossmannLee']), description='perspective function used for variable disaggregation', doc='\n        The perspective function used for variable disaggregation\n\n        "LeeGrossmann" is the original NL convex hull from Lee &\n        Grossmann (2000) [1]_, which substitutes nonlinear constraints\n\n            h_ik(x) <= 0\n\n        with\n\n            x_k = sum( nu_ik )\n            y_ik * h_ik( nu_ik/y_ik ) <= 0\n\n        "GrossmannLee" is an updated formulation from Grossmann &\n        Lee (2003) [2]_, which avoids divide-by-0 errors by using:\n\n            x_k = sum( nu_ik )\n            (y_ik + eps) * h_ik( nu_ik/(y_ik + eps) ) <= 0\n\n        "FurmanSawayaGrossmann" (default) is an improved relaxation [3]_\n        that is exact at 0 and 1 while avoiding numerical issues from\n        the Lee & Grossmann formulation by using:\n\n            x_k = sum( nu_ik )\n            ((1-eps)*y_ik + eps) * h_ik( nu_ik/((1-eps)*y_ik + eps) )                 - eps * h_ki(0) * ( 1-y_ik ) <= 0\n\n        References\n        ----------\n        .. [1] Lee, S., & Grossmann, I. E. (2000). New algorithms for\n           nonlinear generalized disjunctive programming.  Computers and\n           Chemical Engineering, 24, 2125-2141\n\n        .. [2] Grossmann, I. E., & Lee, S. (2003). Generalized disjunctive\n           programming: Nonlinear convex hull relaxation and algorithms.\n           Computational Optimization and Applications, 26, 83-100.\n\n        .. [3] Furman, K., Sawaya, N., and Grossmann, I.  A computationally\n           useful algebraic representation of nonlinear disjunctive convex\n           sets using the perspective function.  Optimization Online\n           (2016). http://www.optimization-online.org/DB_HTML/2016/07/5544.html.\n        '))
    CONFIG.declare('EPS', cfg.ConfigValue(default=0.0001, domain=cfg.PositiveFloat, description='Epsilon value to use in perspective function'))
    CONFIG.declare('assume_fixed_vars_permanent', cfg.ConfigValue(default=False, domain=bool, description='Boolean indicating whether or not to transform so that the transformed model will still be valid when fixed Vars are unfixed.', doc='\n        If True, the transformation will not disaggregate fixed variables.\n        This means that if a fixed variable is unfixed after transformation,\n        the transformed model is no longer valid. By default, the transformation\n        will disagregate fixed variables so that any later fixing and unfixing\n        will be valid in the transformed model.\n        '))
    transformation_name = 'hull'

    def __init__(self):
        super().__init__(logger)
        self._targets = set()

    def _collect_local_vars_from_block(self, block, local_var_dict):
        localVars = block.component('LocalVars')
        if localVars is not None and localVars.ctype is Suffix:
            for disj, var_list in localVars.items():
                local_var_dict[disj].update(var_list)

    def _get_user_defined_local_vars(self, targets):
        user_defined_local_vars = defaultdict(ComponentSet)
        seen_blocks = set()
        for t in targets:
            if t.ctype is Disjunct:
                for b in t.component_data_objects(Block, descend_into=Block, active=True, sort=SortComponents.deterministic):
                    if b not in seen_blocks:
                        self._collect_local_vars_from_block(b, user_defined_local_vars)
                        seen_blocks.add(b)
                blk = t
                while blk is not None:
                    if blk in seen_blocks:
                        break
                    self._collect_local_vars_from_block(blk, user_defined_local_vars)
                    seen_blocks.add(blk)
                    blk = blk.parent_block()
        return user_defined_local_vars

    def _apply_to(self, instance, **kwds):
        try:
            self._apply_to_impl(instance, **kwds)
        finally:
            self._restore_state()
            self._transformation_blocks.clear()
            self._algebraic_constraints.clear()

    def _apply_to_impl(self, instance, **kwds):
        self._process_arguments(instance, **kwds)
        targets = self._filter_targets(instance)
        self._transform_logical_constraints(instance, targets)
        gdp_tree = self._get_gdp_tree_from_targets(instance, targets)
        preprocessed_targets = gdp_tree.reverse_topological_sort()
        local_vars_by_disjunct = self._get_user_defined_local_vars(preprocessed_targets)
        for t in preprocessed_targets:
            if t.ctype is Disjunction:
                self._transform_disjunctionData(t, t.index(), gdp_tree.parent(t), local_vars_by_disjunct)

    def _add_transformation_block(self, to_block):
        transBlock, new_block = super()._add_transformation_block(to_block)
        if not new_block:
            return (transBlock, new_block)
        transBlock.lbub = Set(initialize=['lb', 'ub', 'eq'])
        transBlock.disaggregationConstraints = Constraint(NonNegativeIntegers)
        transBlock._disaggregationConstraintMap = ComponentMap()
        transBlock._disaggregatedVars = Var(NonNegativeIntegers, dense=False)
        transBlock._boundsConstraints = Constraint(NonNegativeIntegers, transBlock.lbub)
        return (transBlock, True)

    def _transform_disjunctionData(self, obj, index, parent_disjunct, local_vars_by_disjunct):
        if not obj.xor:
            raise GDP_Error("Cannot do hull reformulation for Disjunction '%s' with OR constraint. Must be an XOR!" % obj.name)
        active_disjuncts = [disj for disj in obj.disjuncts if disj.active]
        transBlock, xorConstraint = self._setup_transform_disjunctionData(obj, root_disjunct=None)
        disaggregationConstraint = transBlock.disaggregationConstraints
        disaggregationConstraintMap = transBlock._disaggregationConstraintMap
        disaggregatedVars = transBlock._disaggregatedVars
        disaggregated_var_bounds = transBlock._boundsConstraints
        var_order = ComponentSet()
        disjuncts_var_appears_in = ComponentMap()
        disjunct_disaggregated_var_map = {}
        for disjunct in active_disjuncts:
            disjunct_disaggregated_var_map[disjunct] = ComponentMap()
            for var in get_vars_from_components(disjunct, Constraint, include_fixed=not self._config.assume_fixed_vars_permanent, active=True, sort=SortComponents.deterministic, descend_into=Block):
                if var not in var_order:
                    var_order.add(var)
                    disjuncts_var_appears_in[var] = ComponentSet([disjunct])
                else:
                    disjuncts_var_appears_in[var].add(disjunct)
        vars_to_disaggregate = {disj: ComponentSet() for disj in obj.disjuncts}
        all_vars_to_disaggregate = ComponentSet()
        local_vars = defaultdict(ComponentSet)
        for var in var_order:
            disjuncts = disjuncts_var_appears_in[var]
            if len(disjuncts) > 1:
                if self._generate_debug_messages:
                    logger.debug("Assuming '%s' is not a local var since it isused in multiple disjuncts." % var.name)
                for disj in disjuncts:
                    vars_to_disaggregate[disj].add(var)
                    all_vars_to_disaggregate.add(var)
            else:
                disjunct = next(iter(disjuncts))
                if disjunct in local_vars_by_disjunct:
                    if var in local_vars_by_disjunct[disjunct]:
                        local_vars[disjunct].add(var)
                        continue
                vars_to_disaggregate[disjunct].add(var)
                all_vars_to_disaggregate.add(var)
        parent_local_var_list = self._get_local_var_list(parent_disjunct)
        or_expr = 0
        for disjunct in obj.disjuncts:
            or_expr += disjunct.indicator_var.get_associated_binary()
            if disjunct.active:
                self._transform_disjunct(obj=disjunct, transBlock=transBlock, vars_to_disaggregate=vars_to_disaggregate[disjunct], local_vars=local_vars[disjunct], parent_local_var_suffix=parent_local_var_list, parent_disjunct_local_vars=local_vars_by_disjunct[parent_disjunct], disjunct_disaggregated_var_map=disjunct_disaggregated_var_map)
        xorConstraint.add(index, (or_expr, 1))
        obj._algebraic_constraint = weakref_ref(xorConstraint[index])
        for var in all_vars_to_disaggregate:
            if len(disjuncts_var_appears_in[var]) < len(active_disjuncts):
                idx = len(disaggregatedVars)
                disaggregated_var = disaggregatedVars[idx]
                if parent_local_var_list is not None:
                    parent_local_var_list.append(disaggregated_var)
                local_vars_by_disjunct[parent_disjunct].add(disaggregated_var)
                var_free = 1 - sum((disj.indicator_var.get_associated_binary() for disj in disjuncts_var_appears_in[var]))
                self._declare_disaggregated_var_bounds(original_var=var, disaggregatedVar=disaggregated_var, disjunct=obj, bigmConstraint=disaggregated_var_bounds, lb_idx=(idx, 'lb'), ub_idx=(idx, 'ub'), var_free_indicator=var_free)
                var_info = var.parent_block().private_data()
                disaggregated_var_map = var_info.disaggregated_var_map
                dis_var_info = disaggregated_var.parent_block().private_data()
                dis_var_info.bigm_constraint_map[disaggregated_var][obj] = Reference(disaggregated_var_bounds[idx, :])
                dis_var_info.original_var_map[disaggregated_var] = var
                for disj in active_disjuncts:
                    if disj._transformation_block is not None and disj not in disjuncts_var_appears_in[var]:
                        disaggregated_var_map[disj][var] = disaggregated_var
                disaggregatedExpr = disaggregated_var
            else:
                disaggregatedExpr = 0
            for disjunct in disjuncts_var_appears_in[var]:
                disaggregatedExpr += disjunct_disaggregated_var_map[disjunct][var]
            cons_idx = len(disaggregationConstraint)
            disaggregationConstraint.add(cons_idx, var == disaggregatedExpr)
            if var in disaggregationConstraintMap:
                disaggregationConstraintMap[var][obj] = disaggregationConstraint[cons_idx]
            else:
                thismap = disaggregationConstraintMap[var] = ComponentMap()
                thismap[obj] = disaggregationConstraint[cons_idx]
        obj.deactivate()

    def _transform_disjunct(self, obj, transBlock, vars_to_disaggregate, local_vars, parent_local_var_suffix, parent_disjunct_local_vars, disjunct_disaggregated_var_map):
        relaxationBlock = self._get_disjunct_transformation_block(obj, transBlock)
        relaxationBlock.disaggregatedVars = Block()
        for var in vars_to_disaggregate:
            disaggregatedVar = Var(within=Reals, initialize=var.value)
            disaggregatedVarName = unique_component_name(relaxationBlock.disaggregatedVars, var.getname(fully_qualified=True))
            relaxationBlock.disaggregatedVars.add_component(disaggregatedVarName, disaggregatedVar)
            if parent_local_var_suffix is not None:
                parent_local_var_suffix.append(disaggregatedVar)
            parent_disjunct_local_vars.add(disaggregatedVar)
            bigmConstraint = Constraint(transBlock.lbub)
            relaxationBlock.add_component(disaggregatedVarName + '_bounds', bigmConstraint)
            self._declare_disaggregated_var_bounds(original_var=var, disaggregatedVar=disaggregatedVar, disjunct=obj, bigmConstraint=bigmConstraint, lb_idx='lb', ub_idx='ub', var_free_indicator=obj.indicator_var.get_associated_binary())
            data_dict = disaggregatedVar.parent_block().private_data()
            data_dict.bigm_constraint_map[disaggregatedVar][obj] = bigmConstraint
            disjunct_disaggregated_var_map[obj][var] = disaggregatedVar
        for var in local_vars:
            conName = unique_component_name(relaxationBlock, var.getname(fully_qualified=False) + '_bounds')
            bigmConstraint = Constraint(transBlock.lbub)
            relaxationBlock.add_component(conName, bigmConstraint)
            parent_block = var.parent_block()
            self._declare_disaggregated_var_bounds(original_var=var, disaggregatedVar=var, disjunct=obj, bigmConstraint=bigmConstraint, lb_idx='lb', ub_idx='ub', var_free_indicator=obj.indicator_var.get_associated_binary())
            data_dict = var.parent_block().private_data()
            data_dict.bigm_constraint_map[var][obj] = bigmConstraint
            disjunct_disaggregated_var_map[obj][var] = var
        var_substitute_map = dict(((id(v), newV) for v, newV in disjunct_disaggregated_var_map[obj].items()))
        zero_substitute_map = dict(((id(v), ZeroConstant) for v, newV in disjunct_disaggregated_var_map[obj].items()))
        self._transform_block_components(obj, obj, var_substitute_map, zero_substitute_map)
        parent_disjunct_local_vars.update(local_vars)
        obj._deactivate_without_fixing_indicator()

    def _declare_disaggregated_var_bounds(self, original_var, disaggregatedVar, disjunct, bigmConstraint, lb_idx, ub_idx, var_free_indicator):
        lb = original_var.lb
        ub = original_var.ub
        if lb is None or ub is None:
            raise GDP_Error('Variables that appear in disjuncts must be bounded in order to use the hull transformation! Missing bound for %s.' % original_var.name)
        disaggregatedVar.setlb(min(0, lb))
        disaggregatedVar.setub(max(0, ub))
        if lb:
            bigmConstraint.add(lb_idx, var_free_indicator * lb <= disaggregatedVar)
        if ub:
            bigmConstraint.add(ub_idx, disaggregatedVar <= ub * var_free_indicator)
        original_var_info = original_var.parent_block().private_data()
        disaggregated_var_map = original_var_info.disaggregated_var_map
        disaggregated_var_info = disaggregatedVar.parent_block().private_data()
        disaggregated_var_map[disjunct][original_var] = disaggregatedVar
        disaggregated_var_info.original_var_map[disaggregatedVar] = original_var

    def _get_local_var_list(self, parent_disjunct):
        local_var_list = None
        if parent_disjunct is not None:
            self._get_local_var_suffix(parent_disjunct)
            if parent_disjunct.LocalVars.get(parent_disjunct) is None:
                parent_disjunct.LocalVars[parent_disjunct] = []
            local_var_list = parent_disjunct.LocalVars[parent_disjunct]
        return local_var_list

    def _transform_constraint(self, obj, disjunct, var_substitute_map, zero_substitute_map):
        relaxationBlock = disjunct._transformation_block()
        constraintMap = relaxationBlock._constraintMap
        newConstraint = relaxationBlock.transformedConstraints
        for i in sorted(obj.keys()):
            c = obj[i]
            if not c.active:
                continue
            unique = len(newConstraint)
            name = c.local_name + '_%s' % unique
            NL = c.body.polynomial_degree() not in (0, 1)
            EPS = self._config.EPS
            mode = self._config.perspective_function
            if not NL or mode == 'FurmanSawayaGrossmann':
                h_0 = clone_without_expression_components(c.body, substitute=zero_substitute_map)
            y = disjunct.binary_indicator_var
            if NL:
                if mode == 'LeeGrossmann':
                    sub_expr = clone_without_expression_components(c.body, substitute=dict(((var, subs / y) for var, subs in var_substitute_map.items())))
                    expr = sub_expr * y
                elif mode == 'GrossmannLee':
                    sub_expr = clone_without_expression_components(c.body, substitute=dict(((var, subs / (y + EPS)) for var, subs in var_substitute_map.items())))
                    expr = (y + EPS) * sub_expr
                elif mode == 'FurmanSawayaGrossmann':
                    sub_expr = clone_without_expression_components(c.body, substitute=dict(((var, subs / ((1 - EPS) * y + EPS)) for var, subs in var_substitute_map.items())))
                    expr = ((1 - EPS) * y + EPS) * sub_expr - EPS * h_0 * (1 - y)
                else:
                    raise RuntimeError('Unknown NL Hull mode')
            else:
                expr = clone_without_expression_components(c.body, substitute=var_substitute_map)
            if c.equality:
                if NL:
                    newConsExpr = expr == c.lower * y
                else:
                    v = list(EXPR.identify_variables(expr))
                    if len(v) == 1 and (not c.lower):
                        v[0].fix(0)
                        constraintMap['transformedConstraints'][c] = [v[0]]
                        constraintMap['srcConstraints'][v[0]] = c
                        continue
                    newConsExpr = expr - (1 - y) * h_0 == c.lower * y
                if obj.is_indexed():
                    newConstraint.add((name, i, 'eq'), newConsExpr)
                    constraintMap['transformedConstraints'][c] = [newConstraint[name, i, 'eq']]
                    constraintMap['srcConstraints'][newConstraint[name, i, 'eq']] = c
                else:
                    newConstraint.add((name, 'eq'), newConsExpr)
                    constraintMap['transformedConstraints'][c] = [newConstraint[name, 'eq']]
                    constraintMap['srcConstraints'][newConstraint[name, 'eq']] = c
                continue
            if c.lower is not None:
                if self._generate_debug_messages:
                    _name = c.getname(fully_qualified=True)
                    logger.debug('GDP(Hull): Transforming constraint ' + "'%s'", _name)
                if NL:
                    newConsExpr = expr >= c.lower * y
                else:
                    newConsExpr = expr - (1 - y) * h_0 >= c.lower * y
                if obj.is_indexed():
                    newConstraint.add((name, i, 'lb'), newConsExpr)
                    constraintMap['transformedConstraints'][c] = [newConstraint[name, i, 'lb']]
                    constraintMap['srcConstraints'][newConstraint[name, i, 'lb']] = c
                else:
                    newConstraint.add((name, 'lb'), newConsExpr)
                    constraintMap['transformedConstraints'][c] = [newConstraint[name, 'lb']]
                    constraintMap['srcConstraints'][newConstraint[name, 'lb']] = c
            if c.upper is not None:
                if self._generate_debug_messages:
                    _name = c.getname(fully_qualified=True)
                    logger.debug('GDP(Hull): Transforming constraint ' + "'%s'", _name)
                if NL:
                    newConsExpr = expr <= c.upper * y
                else:
                    newConsExpr = expr - (1 - y) * h_0 <= c.upper * y
                if obj.is_indexed():
                    newConstraint.add((name, i, 'ub'), newConsExpr)
                    transformed = constraintMap['transformedConstraints'].get(c)
                    if transformed is not None:
                        transformed.append(newConstraint[name, i, 'ub'])
                    else:
                        constraintMap['transformedConstraints'][c] = [newConstraint[name, i, 'ub']]
                    constraintMap['srcConstraints'][newConstraint[name, i, 'ub']] = c
                else:
                    newConstraint.add((name, 'ub'), newConsExpr)
                    transformed = constraintMap['transformedConstraints'].get(c)
                    if transformed is not None:
                        transformed.append(newConstraint[name, 'ub'])
                    else:
                        constraintMap['transformedConstraints'][c] = [newConstraint[name, 'ub']]
                    constraintMap['srcConstraints'][newConstraint[name, 'ub']] = c
        obj.deactivate()

    def _get_local_var_suffix(self, disjunct):
        localSuffix = disjunct.component('LocalVars')
        if localSuffix is None:
            disjunct.LocalVars = Suffix(direction=Suffix.LOCAL)
        else:
            if localSuffix.ctype is Suffix:
                return
            raise GDP_Error("A component called 'LocalVars' is declared on Disjunct %s, but it is of type %s, not Suffix." % (disjunct.getname(fully_qualified=True), localSuffix.ctype))

    def get_disaggregated_var(self, v, disjunct, raise_exception=True):
        """
        Returns the disaggregated variable corresponding to the Var v and the
        Disjunct disjunct.

        If v is a local variable, this method will return v.

        Parameters
        ----------
        v: a Var that appears in a constraint in a transformed Disjunct
        disjunct: a transformed Disjunct in which v appears
        """
        if disjunct._transformation_block is None:
            raise GDP_Error("Disjunct '%s' has not been transformed" % disjunct.name)
        msg = "It does not appear '%s' is a variable that appears in disjunct '%s'" % (v.name, disjunct.name)
        disaggregated_var_map = v.parent_block().private_data().disaggregated_var_map
        if v in disaggregated_var_map[disjunct]:
            return disaggregated_var_map[disjunct][v]
        elif raise_exception:
            raise GDP_Error(msg)

    def get_src_var(self, disaggregated_var):
        """
        Returns the original model variable to which disaggregated_var
        corresponds.

        Parameters
        ----------
        disaggregated_var: a Var that was created by the hull
                           transformation as a disaggregated variable
                           (and so appears on a transformation block
                           of some Disjunct)
        """
        var_map = disaggregated_var.parent_block().private_data()
        if disaggregated_var in var_map.original_var_map:
            return var_map.original_var_map[disaggregated_var]
        raise GDP_Error("'%s' does not appear to be a disaggregated variable" % disaggregated_var.name)

    def get_disaggregation_constraint(self, original_var, disjunction, raise_exception=True):
        """
        Returns the disaggregation (re-aggregation?) constraint
        (which links the disaggregated variables to their original)
        corresponding to original_var and the transformation of disjunction.

        Parameters
        ----------
        original_var: a Var which was disaggregated in the transformation
                      of Disjunction disjunction
        disjunction: a transformed Disjunction containing original_var
        """
        for disjunct in disjunction.disjuncts:
            transBlock = disjunct.transformation_block
            if transBlock is not None:
                break
        if transBlock is None:
            raise GDP_Error("Disjunction '%s' has not been properly transformed: None of its disjuncts are transformed." % disjunction.name)
        try:
            cons = transBlock.parent_block()._disaggregationConstraintMap[original_var][disjunction]
        except:
            if raise_exception:
                logger.error("It doesn't appear that '%s' is a variable that was disaggregated by Disjunction '%s'" % (original_var.name, disjunction.name))
                raise
            return None
        while not cons.active:
            cons = self.get_transformed_constraints(cons)[0]
        return cons

    def get_var_bounds_constraint(self, v, disjunct=None):
        """
        Returns the IndexedConstraint which sets a disaggregated
        variable to be within its bounds when its Disjunct is active and to
        be 0 otherwise. (It is always an IndexedConstraint because each
        bound becomes a separate constraint.)

        Parameters
        ----------
        v: a Var that was created by the hull transformation as a
           disaggregated variable (and so appears on a transformation
           block of some Disjunct)
        disjunct: (For nested Disjunctions) Which Disjunct in the
           hierarchy the bounds Constraint should correspond to.
           Optional since for non-nested models this can be inferred.
        """
        info = v.parent_block().private_data()
        if v in info.bigm_constraint_map:
            if len(info.bigm_constraint_map[v]) == 1:
                return list(info.bigm_constraint_map[v].values())[0]
            elif disjunct is not None:
                return info.bigm_constraint_map[v][disjunct]
            else:
                raise ValueError("It appears that the variable '%s' appears within a nested GDP hierarchy, and no 'disjunct' argument was specified. Please specify for which Disjunct the bounds constraint for '%s' should be returned." % (v, v))
        raise GDP_Error("Either '%s' is not a disaggregated variable, or the disjunction that disaggregates it has not been properly transformed." % v.name)

    def get_transformed_constraints(self, cons):
        cons = super().get_transformed_constraints(cons)
        while not cons[0].active:
            transformed_cons = []
            for con in cons:
                transformed_cons += super().get_transformed_constraints(con)
            cons = transformed_cons
        return cons