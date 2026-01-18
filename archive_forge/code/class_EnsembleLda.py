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
class EnsembleLda(SaveLoad):
    """Ensemble Latent Dirichlet Allocation (eLDA), a method of training a topic model ensemble.

    Extracts stable topics that are consistently learned across multiple LDA models. eLDA has the added benefit that
    the user does not need to know the exact number of topics the topic model should extract ahead of time.

    """

    def __init__(self, topic_model_class='ldamulticore', num_models=3, min_cores=None, epsilon=0.1, ensemble_workers=1, memory_friendly_ttda=True, min_samples=None, masking_method=mass_masking, masking_threshold=None, distance_workers=1, random_state=None, **gensim_kw_args):
        """Create and train a new EnsembleLda model.

        Will start training immediatelly, except if iterations, passes or num_models is 0 or if the corpus is missing.

        Parameters
        ----------
        topic_model_class : str, topic model, optional
            Examples:
                * 'ldamulticore' (default, recommended)
                * 'lda'
                * ldamodel.LdaModel
                * ldamulticore.LdaMulticore
        ensemble_workers : int, optional
            Spawns that many processes and distributes the models from the ensemble to those as evenly as possible.
            num_models should be a multiple of ensemble_workers.

            Setting it to 0 or 1 will both use the non-multiprocessing version. Default: 1
        num_models : int, optional
            How many LDA models to train in this ensemble.
            Default: 3
        min_cores : int, optional
            Minimum cores a cluster of topics has to contain so that it is recognized as stable topic.
        epsilon : float, optional
            Defaults to 0.1. Epsilon for the CBDBSCAN clustering that generates the stable topics.
        ensemble_workers : int, optional
            Spawns that many processes and distributes the models from the ensemble to those as evenly as possible.
            num_models should be a multiple of ensemble_workers.

            Setting it to 0 or 1 will both use the nonmultiprocessing version. Default: 1
        memory_friendly_ttda : boolean, optional
            If True, the models in the ensemble are deleted after training and only a concatenation of each model's
            topic term distribution (called ttda) is kept to save memory.

            Defaults to True. When False, trained models are stored in a list in self.tms, and no models that are not
            of a gensim model type can be added to this ensemble using the add_model function.

            If False, any topic term matrix can be suplied to add_model.
        min_samples : int, optional
            Required int of nearby topics for a topic to be considered as 'core' in the CBDBSCAN clustering.
        masking_method : function, optional
            Choose one of :meth:`~gensim.models.ensemblelda.mass_masking` (default) or
            :meth:`~gensim.models.ensemblelda.rank_masking` (percentile, faster).

            For clustering, distances between topic-term distributions are asymmetric.  In particular, the distance
            (technically a divergence) from distribution A to B is more of a measure of if A is contained in B.  At a
            high level, this involves using distribution A to mask distribution B and then calculating the cosine
            distance between the two.  The masking can be done in two ways:

            1. mass: forms mask by taking the top ranked terms until their cumulative mass reaches the
            'masking_threshold'

            2. rank: forms mask by taking the top ranked terms (by mass) until the 'masking_threshold' is reached.
            For example, a ranking threshold of 0.11 means the top 0.11 terms by weight are used to form a mask.
        masking_threshold : float, optional
            Default: None, which uses ``0.95`` for "mass", and ``0.11`` for masking_method "rank".  In general, too
            small a mask threshold leads to inaccurate calculations (no signal) and too big a mask leads to noisy
            distance calculations.  Defaults are often a good sweet spot for this hyperparameter.
        distance_workers : int, optional
            When ``distance_workers`` is ``None``, it defaults to ``os.cpu_count()`` for maximum performance. Default is
            1, which is not multiprocessed. Set to ``> 1`` to enable multiprocessing.
        **gensim_kw_args
            Parameters for each gensim model (e.g. :py:class:`gensim.models.LdaModel`) in the ensemble.

        """
        if 'id2word' not in gensim_kw_args:
            gensim_kw_args['id2word'] = None
        if 'corpus' not in gensim_kw_args:
            gensim_kw_args['corpus'] = None
        if gensim_kw_args['id2word'] is None and (not gensim_kw_args['corpus'] is None):
            logger.warning('no word id mapping provided; initializing from corpus, assuming identity')
            gensim_kw_args['id2word'] = utils.dict_from_corpus(gensim_kw_args['corpus'])
        if gensim_kw_args['id2word'] is None and gensim_kw_args['corpus'] is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality. Corpus should be provided using the `corpus` keyword argument.')
        if type(topic_model_class) == type and issubclass(topic_model_class, ldamodel.LdaModel):
            self.topic_model_class = topic_model_class
        else:
            kinds = {'lda': ldamodel.LdaModel, 'ldamulticore': ldamulticore.LdaMulticore}
            if topic_model_class not in kinds:
                raise ValueError("topic_model_class should be one of 'lda', 'ldamulticode' or a model inheriting from LdaModel")
            self.topic_model_class = kinds[topic_model_class]
        self.num_models = num_models
        self.gensim_kw_args = gensim_kw_args
        self.memory_friendly_ttda = memory_friendly_ttda
        self.distance_workers = distance_workers
        self.masking_threshold = masking_threshold
        self.masking_method = masking_method
        self.classic_model_representation = None
        self.random_state = utils.get_random_state(random_state)
        self.sstats_sum = 0
        self.eta = None
        self.tms = []
        self.ttda = np.empty((0, len(gensim_kw_args['id2word'])))
        self.asymmetric_distance_matrix_outdated = True
        if num_models <= 0:
            return
        if gensim_kw_args.get('corpus') is None:
            return
        if 'iterations' in gensim_kw_args and gensim_kw_args['iterations'] <= 0:
            return
        if 'passes' in gensim_kw_args and gensim_kw_args['passes'] <= 0:
            return
        logger.info(f'generating {num_models} topic models using {ensemble_workers} workers')
        if ensemble_workers > 1:
            _generate_topic_models_multiproc(self, num_models, ensemble_workers)
        else:
            _generate_topic_models(self, num_models)
        self._generate_asymmetric_distance_matrix()
        self._generate_topic_clusters(epsilon, min_samples)
        self._generate_stable_topics(min_cores)
        self.generate_gensim_representation()

    def get_topic_model_class(self):
        """Get the class that is used for :meth:`gensim.models.EnsembleLda.generate_gensim_representation`."""
        if self.topic_model_class is None:
            instruction = 'Try setting topic_model_class manually to what the individual models were based on, e.g. LdaMulticore.'
            try:
                module = importlib.import_module(self.topic_model_module_string)
                self.topic_model_class = getattr(module, self.topic_model_class_string)
                del self.topic_model_module_string
                del self.topic_model_class_string
            except ModuleNotFoundError:
                logger.error(f'Could not import the "{self.topic_model_class_string}" module in order to provide the "{self.topic_model_class_string}" class as "topic_model_class" attribute. {instruction}')
            except AttributeError:
                logger.error(f'Could not import the "{self.topic_model_class_string}" class from the "{self.topic_model_module_string}" module in order to set the "topic_model_class" attribute. {instruction}')
        return self.topic_model_class

    def save(self, *args, **kwargs):
        if self.get_topic_model_class() is not None:
            self.topic_model_module_string = self.topic_model_class.__module__
            self.topic_model_class_string = self.topic_model_class.__name__
        kwargs['ignore'] = frozenset(kwargs.get('ignore', ())).union(('topic_model_class',))
        super(EnsembleLda, self).save(*args, **kwargs)
    save.__doc__ = SaveLoad.save.__doc__

    def convert_to_memory_friendly(self):
        """Remove the stored gensim models and only keep their ttdas.

        This frees up memory, but you won't have access to the individual  models anymore if you intended to use them
        outside of the ensemble.
        """
        self.tms = []
        self.memory_friendly_ttda = True

    def generate_gensim_representation(self):
        """Create a gensim model from the stable topics.

        The returned representation is an Gensim LdaModel (:py:class:`gensim.models.LdaModel`) that has been
        instantiated with an A-priori belief on word probability, eta, that represents the topic-term distributions of
        any stable topics the were found by clustering over the ensemble of topic distributions.

        When no stable topics have been detected, None is returned.

        Returns
        -------
        :py:class:`gensim.models.LdaModel`
            A Gensim LDA Model classic_model_representation for which:
            ``classic_model_representation.get_topics() == self.get_topics()``

        """
        logger.info('generating classic gensim model representation based on results from the ensemble')
        sstats_sum = self.sstats_sum
        if sstats_sum == 0 and 'corpus' in self.gensim_kw_args and (not self.gensim_kw_args['corpus'] is None):
            for document in self.gensim_kw_args['corpus']:
                for token in document:
                    sstats_sum += token[1]
            self.sstats_sum = sstats_sum
        stable_topics = self.get_topics()
        num_stable_topics = len(stable_topics)
        if num_stable_topics == 0:
            logger.error('the model did not detect any stable topic. You can try to adjust epsilon: recluster(eps=...)')
            self.classic_model_representation = None
            return
        params = self.gensim_kw_args.copy()
        params['eta'] = self.eta
        params['num_topics'] = num_stable_topics
        params['passes'] = 0
        classic_model_representation = self.get_topic_model_class()(**params)
        eta = classic_model_representation.eta
        if sstats_sum == 0:
            sstats_sum = classic_model_representation.state.sstats.sum()
            self.sstats_sum = sstats_sum
        eta_sum = 0
        if isinstance(eta, (int, float)):
            eta_sum = [eta * len(stable_topics[0])] * num_stable_topics
        else:
            if len(eta.shape) == 1:
                eta_sum = [[eta.sum()]] * num_stable_topics
            if len(eta.shape) > 1:
                eta_sum = np.array(eta.sum(axis=1)[:, None])
        normalization_factor = np.array([[sstats_sum / num_stable_topics]] * num_stable_topics) + eta_sum
        sstats = stable_topics * normalization_factor
        sstats -= eta
        classic_model_representation.state.sstats = sstats.astype(np.float32)
        classic_model_representation.sync_state()
        self.classic_model_representation = classic_model_representation
        return classic_model_representation

    def add_model(self, target, num_new_models=None):
        """Add the topic term distribution array (ttda) of another model to the ensemble.

        This way, multiple topic models can be connected to an ensemble manually. Make sure that all the models use
        the exact same dictionary/idword mapping.

        In order to generate new stable topics afterwards, use:
            2. ``self.``:meth:`~gensim.models.ensemblelda.EnsembleLda.recluster`

        The ttda of another ensemble can also be used, in that case set ``num_new_models`` to the ``num_models``
        parameter of the ensemble, that means the number of classic models in the ensemble that generated the ttda.
        This is important, because that information is used to estimate "min_samples" for _generate_topic_clusters.

        If you trained this ensemble in the past with a certain Dictionary that you want to reuse for other
        models, you can get it from: ``self.id2word``.

        Parameters
        ----------
        target : {see description}
            1. A single EnsembleLda object
            2. List of EnsembleLda objects
            3. A single Gensim topic model (e.g. (:py:class:`gensim.models.LdaModel`)
            4. List of Gensim topic models

            if memory_friendly_ttda is True, target can also be:
            5. topic-term-distribution-array

            example: [[0.1, 0.1, 0.8], [...], ...]

            [topic1, topic2, ...]
            with topic being an array of probabilities:
            [token1, token2, ...]

            token probabilities in a single topic sum to one, therefore, all the words sum to len(ttda)

        num_new_models : integer, optional
            the model keeps track of how many models were used in this ensemble. Set higher if ttda contained topics
            from more than one model. Default: None, which takes care of it automatically.

            If target is a 2D-array of float values, it assumes 1.

            If the ensemble has ``memory_friendly_ttda`` set to False, then it will always use the number of models in
            the target parameter.

        """
        if not isinstance(target, (np.ndarray, list)):
            target = np.array([target])
        else:
            target = np.array(target)
            assert len(target) > 0
        if self.memory_friendly_ttda:
            detected_num_models = 0
            ttda = []
            if isinstance(target.dtype.type(), (np.number, float)):
                ttda = target
                detected_num_models = 1
            elif isinstance(target[0], type(self)):
                ttda = np.concatenate([ensemble.ttda for ensemble in target], axis=0)
                detected_num_models = sum([ensemble.num_models for ensemble in target])
            elif isinstance(target[0], basemodel.BaseTopicModel):
                ttda = np.concatenate([model.get_topics() for model in target], axis=0)
                detected_num_models = len(target)
            else:
                raise ValueError(f'target is of unknown type or a list of unknown types: {type(target[0])}')
            if num_new_models is None:
                self.num_models += detected_num_models
            else:
                self.num_models += num_new_models
        else:
            ttda = []
            if isinstance(target.dtype.type(), (np.number, float)):
                raise ValueError('ttda arrays cannot be added to ensembles, for which memory_friendly_ttda=False, you can call convert_to_memory_friendly, but it will discard the stored gensim models and only keep the relevant topic term distributions from them.')
            elif isinstance(target[0], type(self)):
                for ensemble in target:
                    self.tms += ensemble.tms
                ttda = np.concatenate([ensemble.ttda for ensemble in target], axis=0)
            elif isinstance(target[0], basemodel.BaseTopicModel):
                self.tms += target.tolist()
                ttda = np.concatenate([model.get_topics() for model in target], axis=0)
            else:
                raise ValueError(f'target is of unknown type or a list of unknown types: {type(target[0])}')
            if num_new_models is not None and num_new_models + self.num_models != len(self.tms):
                logger.info('num_new_models will be ignored. num_models should match the number of stored models for a memory unfriendly ensemble')
            self.num_models = len(self.tms)
        logger.info(f'ensemble contains {self.num_models} models and {len(self.ttda)} topics now')
        if self.ttda.shape[1] != ttda.shape[1]:
            raise ValueError(f'target ttda dimensions do not match. Topics must be {self.ttda.shape[-1]} but was {ttda.shape[-1]} elements large')
        self.ttda = np.append(self.ttda, ttda, axis=0)
        self.asymmetric_distance_matrix_outdated = True

    def _generate_asymmetric_distance_matrix(self):
        """Calculate the pairwise distance matrix for all the ttdas from the ensemble.

        Returns the asymmetric pairwise distance matrix that is used in the DBSCAN clustering.

        Afterwards, the model needs to be reclustered for this generated matrix to take effect.

        """
        workers = self.distance_workers
        self.asymmetric_distance_matrix_outdated = False
        logger.info(f'generating a {len(self.ttda)} x {len(self.ttda)} asymmetric distance matrix...')
        if workers is not None and workers <= 1:
            self.asymmetric_distance_matrix = _calculate_asymmetric_distance_matrix_chunk(ttda1=self.ttda, ttda2=self.ttda, start_index=0, masking_method=self.masking_method, masking_threshold=self.masking_threshold)
        else:
            if workers is None:
                workers = os.cpu_count()
            self.asymmetric_distance_matrix = _calculate_assymetric_distance_matrix_multiproc(workers=workers, entire_ttda=self.ttda, masking_method=self.masking_method, masking_threshold=self.masking_threshold)

    def _generate_topic_clusters(self, eps=0.1, min_samples=None):
        """Run the CBDBSCAN algorithm on all the detected topics and label them with label-indices.

        The final approval and generation of stable topics is done in ``_generate_stable_topics()``.

        Parameters
        ----------
        eps : float
            dbscan distance scale
        min_samples : int, optional
            defaults to ``int(self.num_models / 2)``, dbscan min neighbours threshold required to consider
            a topic to be a core. Should scale with the number of models, ``self.num_models``

        """
        if min_samples is None:
            min_samples = int(self.num_models / 2)
            logger.info('fitting the clustering model, using %s for min_samples', min_samples)
        else:
            logger.info('fitting the clustering model')
        self.cluster_model = CBDBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_model.fit(self.asymmetric_distance_matrix)

    def _generate_stable_topics(self, min_cores=None):
        """Generate stable topics out of the clusters.

        The function finds clusters of topics using a variant of DBScan.  If a cluster has enough core topics
        (c.f. parameter ``min_cores``), then this cluster represents a stable topic.  The stable topic is specifically
        calculated as the average over all topic-term distributions of the core topics in the cluster.

        This function is the last step that has to be done in the ensemble.  After this step is complete,
        Stable topics can be retrieved afterwards using the :meth:`~gensim.models.ensemblelda.EnsembleLda.get_topics`
        method.

        Parameters
        ----------
        min_cores : int
            Minimum number of core topics needed to form a cluster that represents a stable topic.
                Using ``None`` defaults to ``min_cores = min(3, max(1, int(self.num_models /4 +1)))``

        """
        if min_cores == 0:
            min_cores = 1
        if min_cores is None:
            min_cores = min(3, max(1, int(self.num_models / 4 + 1)))
            logger.info('generating stable topics, using %s for min_cores', min_cores)
        else:
            logger.info('generating stable topics')
        cbdbscan_topics = self.cluster_model.results
        grouped_by_labels = _group_by_labels(cbdbscan_topics)
        clusters = _aggregate_topics(grouped_by_labels)
        valid_clusters = _validate_clusters(clusters, min_cores)
        valid_cluster_labels = {cluster.label for cluster in valid_clusters}
        for topic in cbdbscan_topics:
            topic.valid_neighboring_labels = {label for label in topic.neighboring_labels if label in valid_cluster_labels}
        valid_core_mask = np.vectorize(_is_valid_core)(cbdbscan_topics)
        valid_topics = self.ttda[valid_core_mask]
        topic_labels = np.array([topic.label for topic in cbdbscan_topics])[valid_core_mask]
        unique_labels = np.unique(topic_labels)
        num_stable_topics = len(unique_labels)
        stable_topics = np.empty((num_stable_topics, len(self.id2word)))
        for label_index, label in enumerate(unique_labels):
            topics_of_cluster = np.array([topic for t, topic in enumerate(valid_topics) if topic_labels[t] == label])
            stable_topics[label_index] = topics_of_cluster.mean(axis=0)
        self.valid_clusters = valid_clusters
        self.stable_topics = stable_topics
        logger.info('found %s stable topics', len(stable_topics))

    def recluster(self, eps=0.1, min_samples=None, min_cores=None):
        """Reapply CBDBSCAN clustering and stable topic generation.

        Stable topics can be retrieved using :meth:`~gensim.models.ensemblelda.EnsembleLda.get_topics`.

        Parameters
        ----------
        eps : float
            epsilon for the CBDBSCAN algorithm, having the same meaning as in classic DBSCAN clustering.
            default: ``0.1``
        min_samples : int
            The minimum number of samples in the neighborhood of a topic to be considered a core in CBDBSCAN.
            default: ``int(self.num_models / 2)``
        min_cores : int
            how many cores a cluster has to have, to be treated as stable topic. That means, how many topics
            that look similar have to be present, so that the average topic in those is used as stable topic.
            default: ``min(3, max(1, int(self.num_models /4 +1)))``

        """
        if self.asymmetric_distance_matrix_outdated:
            logger.info('asymmetric distance matrix is outdated due to add_model')
            self._generate_asymmetric_distance_matrix()
        self._generate_topic_clusters(eps, min_samples)
        self._generate_stable_topics(min_cores)
        self.generate_gensim_representation()

    def get_topics(self):
        """Return only the stable topics from the ensemble.

        Returns
        -------
        2D Numpy.numpy.ndarray of floats
            List of stable topic term distributions

        """
        return self.stable_topics

    def _ensure_gensim_representation(self):
        """Check if stable topics and the internal gensim representation exist. Raise an error if not."""
        if self.classic_model_representation is None:
            if len(self.stable_topics) == 0:
                raise ValueError('no stable topic was detected')
            else:
                raise ValueError('use generate_gensim_representation() first')

    def __getitem__(self, i):
        """See :meth:`gensim.models.LdaModel.__getitem__`."""
        self._ensure_gensim_representation()
        return self.classic_model_representation[i]

    def inference(self, *posargs, **kwargs):
        """See :meth:`gensim.models.LdaModel.inference`."""
        self._ensure_gensim_representation()
        return self.classic_model_representation.inference(*posargs, **kwargs)

    def log_perplexity(self, *posargs, **kwargs):
        """See :meth:`gensim.models.LdaModel.log_perplexity`."""
        self._ensure_gensim_representation()
        return self.classic_model_representation.log_perplexity(*posargs, **kwargs)

    def print_topics(self, *posargs, **kwargs):
        """See :meth:`gensim.models.LdaModel.print_topics`."""
        self._ensure_gensim_representation()
        return self.classic_model_representation.print_topics(*posargs, **kwargs)

    @property
    def id2word(self):
        """Return the :py:class:`gensim.corpora.dictionary.Dictionary` object used in the model."""
        return self.gensim_kw_args['id2word']