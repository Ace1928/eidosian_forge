import operator
import socket
from collections import defaultdict
from copy import deepcopy
from enum import Enum, auto
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from urllib.parse import urlparse
import numpy as np
import scipy.sparse as ss
from .basic import LightGBMError, _choose_param_value, _ConfigAliases, _log_info, _log_warning
from .compat import (DASK_INSTALLED, PANDAS_INSTALLED, SKLEARN_INSTALLED, Client, Future, LGBMNotFittedError, concat,
from .sklearn import (LGBMClassifier, LGBMModel, LGBMRanker, LGBMRegressor, _LGBM_ScikitCustomObjectiveFunction,
def _train_part(params: Dict[str, Any], model_factory: Type[LGBMModel], list_of_parts: List[Dict[str, _DaskPart]], machines: str, local_listen_port: int, num_machines: int, return_model: bool, time_out: int, remote_socket: _RemoteSocket, **kwargs: Any) -> Optional[LGBMModel]:
    network_params = {'machines': machines, 'local_listen_port': local_listen_port, 'time_out': time_out, 'num_machines': num_machines}
    params.update(network_params)
    is_ranker = issubclass(model_factory, LGBMRanker)
    data = _concat([x['data'] for x in list_of_parts])
    label = _concat([x['label'] for x in list_of_parts])
    if 'weight' in list_of_parts[0]:
        weight = _concat([x['weight'] for x in list_of_parts])
    else:
        weight = None
    if 'group' in list_of_parts[0]:
        group = _concat([x['group'] for x in list_of_parts])
    else:
        group = None
    if 'init_score' in list_of_parts[0]:
        init_score = _concat([x['init_score'] for x in list_of_parts])
    else:
        init_score = None
    n_evals = max((len(x.get('eval_set', [])) for x in list_of_parts))
    eval_names = kwargs.pop('eval_names', None)
    eval_class_weight = kwargs.get('eval_class_weight')
    local_eval_set = None
    local_eval_names = None
    local_eval_sample_weight = None
    local_eval_init_score = None
    local_eval_group = None
    if n_evals:
        has_eval_sample_weight = any((x.get('eval_sample_weight') is not None for x in list_of_parts))
        has_eval_init_score = any((x.get('eval_init_score') is not None for x in list_of_parts))
        local_eval_set = []
        evals_result_names = []
        if has_eval_sample_weight:
            local_eval_sample_weight = []
        if has_eval_init_score:
            local_eval_init_score = []
        if is_ranker:
            local_eval_group = []
        missing_eval_component_idx = []
        for i in range(n_evals):
            x_e = []
            y_e = []
            w_e = []
            init_score_e = []
            g_e = []
            for part in list_of_parts:
                if not part.get('eval_set'):
                    continue
                if eval_names:
                    evals_result_name = eval_names[i]
                else:
                    evals_result_name = f'valid_{i}'
                eval_set = part['eval_set'][i]
                if eval_set is _DatasetNames.TRAINSET:
                    x_e.append(part['data'])
                    y_e.append(part['label'])
                else:
                    x_e.extend(eval_set[0])
                    y_e.extend(eval_set[1])
                if evals_result_name not in evals_result_names:
                    evals_result_names.append(evals_result_name)
                eval_weight = part.get('eval_sample_weight')
                if eval_weight:
                    if eval_weight[i] is _DatasetNames.SAMPLE_WEIGHT:
                        w_e.append(part['weight'])
                    else:
                        w_e.extend(eval_weight[i])
                eval_init_score = part.get('eval_init_score')
                if eval_init_score:
                    if eval_init_score[i] is _DatasetNames.INIT_SCORE:
                        init_score_e.append(part['init_score'])
                    else:
                        init_score_e.extend(eval_init_score[i])
                eval_group = part.get('eval_group')
                if eval_group:
                    if eval_group[i] is _DatasetNames.GROUP:
                        g_e.append(part['group'])
                    else:
                        g_e.extend(eval_group[i])
            x_e, y_e, w_e, init_score_e, g_e = _remove_list_padding(x_e, y_e, w_e, init_score_e, g_e)
            if x_e:
                local_eval_set.append((_concat(x_e), _concat(y_e)))
            else:
                missing_eval_component_idx.append(i)
                continue
            if w_e:
                local_eval_sample_weight.append(_concat(w_e))
            if init_score_e:
                local_eval_init_score.append(_concat(init_score_e))
            if g_e:
                local_eval_group.append(_concat(g_e))
        eval_component_idx = [i for i in range(n_evals) if i not in missing_eval_component_idx]
        if eval_names:
            local_eval_names = [eval_names[i] for i in eval_component_idx]
        if eval_class_weight:
            kwargs['eval_class_weight'] = [eval_class_weight[i] for i in eval_component_idx]
    model = model_factory(**params)
    if remote_socket is not None:
        remote_socket.release()
    try:
        if is_ranker:
            model.fit(data, label, sample_weight=weight, init_score=init_score, group=group, eval_set=local_eval_set, eval_sample_weight=local_eval_sample_weight, eval_init_score=local_eval_init_score, eval_group=local_eval_group, eval_names=local_eval_names, **kwargs)
        else:
            model.fit(data, label, sample_weight=weight, init_score=init_score, eval_set=local_eval_set, eval_sample_weight=local_eval_sample_weight, eval_init_score=local_eval_init_score, eval_names=local_eval_names, **kwargs)
    finally:
        if getattr(model, 'fitted_', False):
            model.booster_.free_network()
    if n_evals:
        model = _pad_eval_names(model, required_names=evals_result_names)
    return model if return_model else None