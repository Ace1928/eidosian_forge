import logging
import os
from multiprocessing import Process, Pipe, ProcessError
import importlib
from typing import Set, Optional, List
import numpy as np
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from gensim import utils
from gensim.models import ldamodel, ldamulticore, basemodel
from gensim.utils import SaveLoad
def _generate_topic_models_multiproc(ensemble, num_models, ensemble_workers):
    """Generate the topic models to form the ensemble in a multiprocessed way.

    Depending on the used topic model this can result in a speedup.

    Parameters
    ----------
    ensemble: EnsembleLda
        the ensemble
    num_models : int
        how many models to train in the ensemble
    ensemble_workers : int
        into how many processes to split the models will be set to max(workers, num_models), to avoid workers that
        are supposed to train 0 models.

        to get maximum performance, set to the number of your cores, if non-parallelized models are being used in
        the ensemble (LdaModel).

        For LdaMulticore, the performance gain is small and gets larger for a significantly smaller corpus.
        In that case, ensemble_workers=2 can be used.

    """
    random_states = [ensemble.random_state.randint(_MAX_RANDOM_STATE) for _ in range(num_models)]
    workers = min(ensemble_workers, num_models)
    processes = []
    pipes = []
    num_models_unhandled = num_models
    for i in range(workers):
        parent_conn, child_conn = Pipe()
        num_subprocess_models = 0
        if i == workers - 1:
            num_subprocess_models = num_models_unhandled
        else:
            num_subprocess_models = int(num_models_unhandled / (workers - i))
        random_states_for_worker = random_states[-num_models_unhandled:][:num_subprocess_models]
        args = (ensemble, num_subprocess_models, random_states_for_worker, child_conn)
        try:
            process = Process(target=_generate_topic_models_worker, args=args)
            processes.append(process)
            pipes.append((parent_conn, child_conn))
            process.start()
            num_models_unhandled -= num_subprocess_models
        except ProcessError:
            logger.error(f'could not start process {i}')
            _teardown(pipes, processes)
            raise
    for parent_conn, _ in pipes:
        answer = parent_conn.recv()
        parent_conn.close()
        if not ensemble.memory_friendly_ttda:
            ensemble.tms += answer
            ttda = np.concatenate([m.get_topics() for m in answer])
        else:
            ttda = answer
        ensemble.ttda = np.concatenate([ensemble.ttda, ttda])
    for process in processes:
        process.terminate()