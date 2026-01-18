from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
def _compare_model_topics(self, model_topics):
    """Get average topic and model coherences.

        Parameters
        ----------
        model_topics : list of list of str
            Topics from the model.

        Returns
        -------
        list of (float, float)
            Sequence of pairs of average topic coherence and average coherence.

        """
    coherences = []
    last_topn_value = min(self.topn - 1, 4)
    topn_grid = list(range(self.topn, last_topn_value, -5))
    for model_num, topics in enumerate(model_topics):
        self.topics = topics
        coherence_at_n = {}
        for n in topn_grid:
            self.topn = n
            topic_coherences = self.get_coherence_per_topic()
            filled_coherences = np.array(topic_coherences)
            filled_coherences[np.isnan(filled_coherences)] = np.nanmean(filled_coherences)
            coherence_at_n[n] = (topic_coherences, self.aggregate_measures(filled_coherences))
        topic_coherences, avg_coherences = zip(*coherence_at_n.values())
        avg_topic_coherences = np.vstack(topic_coherences).mean(0)
        model_coherence = np.mean(avg_coherences)
        logging.info('Avg coherence for model %d: %.5f' % (model_num, model_coherence))
        coherences.append((avg_topic_coherences, model_coherence))
    return coherences