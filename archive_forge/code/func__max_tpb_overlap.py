import functools
import itertools
from operator import mul
from typing import Dict, List, Iterable, Sequence, Set, Tuple, Union, cast
import networkx as nx
from networkx.algorithms.approximation.clique import clique_removal
from pyquil.experiment._main import Experiment
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._setting import ExperimentSetting, TensorProductState, _OneQState
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.paulis import PauliTerm, sI
from pyquil.quil import Program
def _max_tpb_overlap(tomo_expt: Experiment) -> Dict[ExperimentSetting, List[ExperimentSetting]]:
    """
    Given an input Experiment, provide a dictionary indicating which ExperimentSettings
    share a tensor product basis

    :param tomo_expt: Experiment, from which to group ExperimentSettings that share a tpb
        and can be run together
    :return: dictionary keyed with ExperimentSetting (specifying a tpb), and with each value being a
            list of ExperimentSettings (diagonal in that tpb)
    """
    diagonal_sets: Dict[ExperimentSetting, List[ExperimentSetting]] = {}
    for expt_setting in tomo_expt:
        assert len(expt_setting) == 1, 'already grouped?'
        unpacked_expt_setting = expt_setting[0]
        found_tpb = False
        for es, es_list in diagonal_sets.items():
            trial_es_list = es_list + [unpacked_expt_setting]
            diag_in_term = _max_weight_state((expst.in_state for expst in trial_es_list))
            diag_out_term = _max_weight_operator((expst.out_operator for expst in trial_es_list))
            if diag_in_term is not None and diag_out_term is not None:
                found_tpb = True
                assert len(diag_in_term) >= len(es.in_state), "Highest weight in-state can't be smaller than the given in-state"
                assert len(diag_out_term) >= len(es.out_operator), "Highest weight out-PauliTerm can't be smaller than the given out-PauliTerm"
                if len(diag_in_term) > len(es.in_state) or len(diag_out_term) > len(es.out_operator):
                    del diagonal_sets[es]
                    new_es = ExperimentSetting(diag_in_term, diag_out_term)
                    diagonal_sets[new_es] = trial_es_list
                else:
                    diagonal_sets[es] = trial_es_list
                break
        if not found_tpb:
            diagonal_sets[unpacked_expt_setting] = [unpacked_expt_setting]
    return diagonal_sets