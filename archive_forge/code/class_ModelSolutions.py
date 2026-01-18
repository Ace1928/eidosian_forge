import logging
import sys
from weakref import ref as weakref_ref
import gc
import math
from pyomo.common import timing
from pyomo.common.collections import Bunch
from pyomo.common.dependencies import pympler, pympler_available
from pyomo.common.deprecation import deprecated
from pyomo.common.gc_manager import PauseGC
from pyomo.common.log import is_debug_set
from pyomo.common.numeric_types import value
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.base.block import ScalarBlock
from pyomo.core.base.set import Set
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.label import CNameLabeler, CuidLabeler
from pyomo.dataportal.DataPortal import DataPortal
from pyomo.opt.results import Solution, SolverStatus, UndefinedData
from contextlib import nullcontext
from io import StringIO
class ModelSolutions(object):

    def __init__(self, instance):
        self._instance = weakref_ref(instance)
        self.clear()

    def clear(self, clear_symbol_maps=True):
        if clear_symbol_maps:
            self.symbol_map = {}
        self.solutions = []
        self.index = None

    def __getstate__(self):
        state = {}
        state['index'] = self.index
        state['_instance'] = self._instance()
        state['solutions'] = self.solutions
        state['symbol_map'] = self.symbol_map
        return state

    def __setstate__(self, state):
        for key, val in state.items():
            setattr(self, key, val)
        self._instance = weakref_ref(self._instance)

    def __len__(self):
        return len(self.solutions)

    def __getitem__(self, index):
        return self.solutions[index]

    def add_symbol_map(self, symbol_map):
        self.symbol_map[id(symbol_map)] = symbol_map

    def delete_symbol_map(self, smap_id):
        if not smap_id is None:
            del self.symbol_map[smap_id]

    def load_from(self, results, allow_consistent_values_for_fixed_vars=False, comparison_tolerance_for_fixed_vars=1e-05, ignore_invalid_labels=False, id=None, delete_symbol_map=True, clear=True, default_variable_value=None, select=0, ignore_fixed_vars=True):
        """
        Load solver results
        """
        instance = self._instance()
        if results.solver.status == SolverStatus.warning:
            tc = getattr(results.solver, 'termination_condition', None)
            msg = getattr(results.solver, 'message', None)
            logger.warning('Loading a SolverResults object with a warning status into model.name="%s";\n  - termination condition: %s\n  - message from solver: %s' % (instance.name, tc, msg))
        elif results.solver.status != SolverStatus.ok:
            if results.solver.status == SolverStatus.aborted and len(results.solution) > 0:
                logger.warning("Loading a SolverResults object with an 'aborted' status, but containing a solution")
            else:
                raise ValueError('Cannot load a SolverResults object with bad status: %s' % str(results.solver.status))
        if clear:
            self.clear(clear_symbol_maps=False)
        if len(results.solution) == 0:
            return
        smap = results.__dict__.get('_smap', None)
        if not smap is None:
            smap_id = id_func(smap)
            self.add_symbol_map(smap)
            results._smap = None
        else:
            smap_id = results.__dict__.get('_smap_id')
        cache = {}
        if not id is None:
            self.add_solution(results.solution(id), smap_id, delete_symbol_map=False, cache=cache, ignore_invalid_labels=ignore_invalid_labels, default_variable_value=default_variable_value)
        else:
            for i in range(len(results.solution)):
                self.add_solution(results.solution(i), smap_id, delete_symbol_map=False, cache=cache, ignore_invalid_labels=ignore_invalid_labels, default_variable_value=default_variable_value)
        if delete_symbol_map:
            self.delete_symbol_map(smap_id)
        if not select is None:
            self.select(select, allow_consistent_values_for_fixed_vars=allow_consistent_values_for_fixed_vars, comparison_tolerance_for_fixed_vars=comparison_tolerance_for_fixed_vars, ignore_invalid_labels=ignore_invalid_labels, ignore_fixed_vars=ignore_fixed_vars)

    def store_to(self, results, cuid=False, skip_stale_vars=False):
        """
        Return a Solution() object that is populated with the values in the model.
        """
        instance = self._instance()
        results.solution.clear()
        results._smap_id = None
        for soln_ in self.solutions:
            soln = Solution()
            soln._cuid = cuid
            for key, val in soln_._metadata.items():
                setattr(soln, key, val)
            if cuid:
                labeler = CuidLabeler()
            else:
                labeler = CNameLabeler()
            sm = SymbolMap()
            entry = soln_._entry['objective']
            for obj in instance.component_data_objects(Objective, active=True):
                vals = entry.get(id(obj), None)
                if vals is None:
                    vals = {}
                else:
                    vals = vals[1]
                vals['Value'] = value(obj)
                soln.objective[sm.getSymbol(obj, labeler)] = vals
            entry = soln_._entry['variable']
            for obj in instance.component_data_objects(Var, active=True):
                if obj.stale and skip_stale_vars:
                    continue
                vals = entry.get(id(obj), None)
                if vals is None:
                    vals = {}
                else:
                    vals = vals[1]
                vals['Value'] = value(obj)
                soln.variable[sm.getSymbol(obj, labeler)] = vals
            entry = soln_._entry['constraint']
            for obj in instance.component_data_objects(Constraint, active=True):
                vals = entry.get(id(obj), None)
                if vals is None:
                    continue
                else:
                    vals = vals[1]
                soln.constraint[sm.getSymbol(obj, labeler)] = vals
            results.solution.insert(soln)

    def add_solution(self, solution, smap_id, delete_symbol_map=True, cache=None, ignore_invalid_labels=False, ignore_missing_symbols=True, default_variable_value=None):
        instance = self._instance()
        soln = ModelSolution()
        soln._metadata['status'] = solution.status
        if not type(solution.message) is UndefinedData:
            soln._metadata['message'] = solution.message
        if not type(solution.gap) is UndefinedData:
            soln._metadata['gap'] = solution.gap
        if smap_id is None:
            if cache is None:
                cache = {}
            if solution._cuid:
                if len(cache) == 0:
                    for obj in instance.component_data_objects(Var):
                        cache[ComponentUID(obj)] = obj
                    for obj in instance.component_data_objects(Objective, active=True):
                        cache[ComponentUID(obj)] = obj
                    for obj in instance.component_data_objects(Constraint, active=True):
                        cache[ComponentUID(obj)] = obj
                for name in ['problem', 'objective', 'variable', 'constraint']:
                    tmp = soln._entry[name]
                    for cuid, val in getattr(solution, name).items():
                        obj = cache.get(cuid, None)
                        if obj is None:
                            if ignore_invalid_labels:
                                continue
                            raise RuntimeError('CUID %s is missing from model %s' % (str(cuid), instance.name))
                        tmp[id(obj)] = (obj, val)
            else:
                if len(cache) == 0:
                    for obj in instance.component_data_objects(Var):
                        cache[obj.name] = obj
                    for obj in instance.component_data_objects(Objective, active=True):
                        cache[obj.name] = obj
                    for obj in instance.component_data_objects(Constraint, active=True):
                        cache[obj.name] = obj
                for name in ['problem', 'objective', 'variable', 'constraint']:
                    tmp = soln._entry[name]
                    for symb, val in getattr(solution, name).items():
                        obj = cache.get(symb, None)
                        if obj is None:
                            if ignore_invalid_labels:
                                continue
                            raise RuntimeError('Symbol %s is missing from model %s' % (symb, instance.name))
                        tmp[id(obj)] = (obj, val)
        else:
            smap = self.symbol_map[smap_id]
            for name in ['problem', 'objective', 'variable', 'constraint']:
                tmp = soln._entry[name]
                for symb, val in getattr(solution, name).items():
                    if symb in smap.bySymbol:
                        obj = smap.bySymbol[symb]
                    elif symb in smap.aliases:
                        obj = smap.aliases[symb]
                    elif ignore_missing_symbols:
                        continue
                    else:
                        raise RuntimeError('ERROR: Symbol %s is missing from model %s when loading with a symbol map!' % (symb, instance.name))
                    tmp[id(obj)] = (obj, val)
            if delete_symbol_map:
                self.delete_symbol_map(smap_id)
        tmp = soln._entry['variable']
        for vdata in instance.component_data_objects(Var):
            id_ = id(vdata)
            if vdata.fixed:
                tmp[id_] = (vdata, {'Value': vdata.value})
            elif default_variable_value is not None and smap_id is not None and (id_ in smap.byObject) and (id_ not in tmp):
                tmp[id_] = (vdata, {'Value': default_variable_value})
        self.solutions.append(soln)
        return len(self.solutions) - 1

    def select(self, index=0, allow_consistent_values_for_fixed_vars=False, comparison_tolerance_for_fixed_vars=1e-05, ignore_invalid_labels=False, ignore_fixed_vars=True):
        """
        Select a solution from the model's solutions.

        allow_consistent_values_for_fixed_vars: a flag that
        indicates whether a solution can specify consistent
        values for variables in the model that are fixed.

        ignore_invalid_labels: a flag that indicates whether
        labels in the solution that don't appear in the model
        yield an error. This allows for loading a results object
        generated from one model into another related, but not
        identical, model.
        """
        instance = self._instance()
        StaleFlagManager.mark_all_as_stale()
        if index is not None:
            self.index = index
        soln = self.solutions[self.index]
        valid_import_suffixes = dict(active_import_suffix_generator(instance))
        for suffix in valid_import_suffixes.values():
            suffix.clear_all_values()
        for id_, (pobj, entry) in soln._entry['problem'].items():
            for _attr_key, attr_value in entry.items():
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][pobj] = attr_value
        for id_, (odata, entry) in soln._entry['objective'].items():
            for _attr_key, attr_value in entry.items():
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][odata] = attr_value
        for id_, (vdata, entry) in soln._entry['variable'].items():
            val = entry['Value']
            if vdata.fixed is True:
                if ignore_fixed_vars:
                    continue
                if not allow_consistent_values_for_fixed_vars:
                    msg = "Variable '%s' in model '%s' is currently fixed - new value is not expected in solution"
                    raise TypeError(msg % (vdata.name, instance.name))
                if math.fabs(val - vdata.value) > comparison_tolerance_for_fixed_vars:
                    raise TypeError("Variable '%s' in model '%s' is currently fixed - a value of '%s' in solution is not within tolerance=%s of the current value of '%s'" % (vdata.name, instance.name, str(val), str(comparison_tolerance_for_fixed_vars), str(vdata.value)))
            vdata.set_value(val, skip_validation=True)
            for _attr_key, attr_value in entry.items():
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key == 'value':
                    continue
                elif attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][vdata] = attr_value
        for id_, (cdata, entry) in soln._entry['constraint'].items():
            for _attr_key, attr_value in entry.items():
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][cdata] = attr_value
        StaleFlagManager.mark_all_as_stale(delayed=True)