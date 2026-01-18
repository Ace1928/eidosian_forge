from pyomo.core import TransformationFactory, Var, NonNegativeReals
from pyomo.core.base.misc import create_name
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.plugins.transform.util import collectAbstractComponents
@TransformationFactory.register('core.add_slack_vars', doc='Create an equivalent model by introducing slack variables to eliminate inequality constraints.')
class EqualityTransform(IsomorphicTransformation):
    """
    Creates a new, equivalent model by introducing slack and excess variables
    to eliminate inequality constraints.
    """

    def __init__(self, **kwds):
        kwds['name'] = kwds.pop('name', 'add_slack_vars')
        super(EqualityTransform, self).__init__(**kwds)

    def _create_using(self, model, **kwds):
        """
        Eliminate inequality constraints.

        Required arguments:

          model The model to transform.

        Optional keyword arguments:

          slack_root  The root name of auxiliary slack variables.
                      Default is 'auxiliary_slack'.
          excess_root The root name of auxiliary slack variables.
                      Default is 'auxiliary_excess'.
          lb_suffix   The suffix applied to converted upper bound constraints
                      Default is '_lower_bound'.
          ub_suffix   The suffix applied to converted lower bound constraints
                      Default is '_upper_bound'.
        """
        slack_suffix = kwds.pop('slack_suffix', 'slack')
        excess_suffix = kwds.pop('excess_suffix', 'excess')
        lb_suffix = kwds.pop('lb_suffix', 'lb')
        ub_suffix = kwds.pop('ub_suffix', 'ub')
        equality = model.clone()
        components = collectAbstractComponents(equality)
        for con_name in components['Constraint']:
            con = equality.__getattribute__(con_name)
            indices = con._data.keys()
            for ndx, cdata in [(ndx, con._data[ndx]) for ndx in indices]:
                qualified_con_name = create_name(con_name, ndx)
                if cdata.equality:
                    continue
                if cdata.lower is not None:
                    excess_name = '%s_%s' % (qualified_con_name, excess_suffix)
                    equality.__setattr__(excess_name, Var(within=NonNegativeReals))
                    lb_name = '%s_%s' % (create_name('', ndx), lb_suffix)
                    excess = equality.__getattribute__(excess_name)
                    new_expr = cdata.lower == cdata.body - excess
                    con.add(lb_name, new_expr)
                if cdata.upper is not None:
                    slack_name = '%s_%s' % (qualified_con_name, slack_suffix)
                    equality.__setattr__(slack_name, Var(within=NonNegativeReals))
                    ub_name = '%s_%s' % (create_name('', ndx), ub_suffix)
                    slack = equality.__getattribute__(slack_name)
                    new_expr = cdata.upper == cdata.body + slack
                    con.add(ub_name, new_expr)
                del con._data[ndx]
        return equality.create()