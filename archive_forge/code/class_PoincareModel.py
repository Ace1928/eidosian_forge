import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter
import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors
class PoincareModel(utils.SaveLoad):
    """Train, use and evaluate Poincare Embeddings.

    The model can be stored/loaded via its :meth:`~gensim.models.poincare.PoincareModel.save`
    and :meth:`~gensim.models.poincare.PoincareModel.load` methods, or stored/loaded in the word2vec format
    via `model.kv.save_word2vec_format` and :meth:`~gensim.models.poincare.PoincareKeyedVectors.load_word2vec_format`.

    Notes
    -----
    Training cannot be resumed from a model loaded via `load_word2vec_format`, if you wish to train further,
    use :meth:`~gensim.models.poincare.PoincareModel.save` and :meth:`~gensim.models.poincare.PoincareModel.load`
    methods instead.

    An important attribute (that provides a lot of additional functionality when directly accessed) are the
    keyed vectors:

    self.kv : :class:`~gensim.models.poincare.PoincareKeyedVectors`
        This object essentially contains the mapping between nodes and embeddings, as well the vocabulary of the model
        (set of unique nodes seen by the model). After training, it can be used to perform operations on the vectors
        such as vector lookup, distance and similarity calculations etc.
        See the documentation of its class for usage examples.

    """

    def __init__(self, train_data, size=50, alpha=0.1, negative=10, workers=1, epsilon=1e-05, regularization_coeff=1.0, burn_in=10, burn_in_alpha=0.01, init_range=(-0.001, 0.001), dtype=np.float64, seed=0):
        """Initialize and train a Poincare embedding model from an iterable of relations.

        Parameters
        ----------
        train_data : {iterable of (str, str), :class:`gensim.models.poincare.PoincareRelations`}
            Iterable of relations, e.g. a list of tuples, or a :class:`gensim.models.poincare.PoincareRelations`
            instance streaming from a file. Note that the relations are treated as ordered pairs,
            i.e. a relation (a, b) does not imply the opposite relation (b, a). In case the relations are symmetric,
            the data should contain both relations (a, b) and (b, a).
        size : int, optional
            Number of dimensions of the trained model.
        alpha : float, optional
            Learning rate for training.
        negative : int, optional
            Number of negative samples to use.
        workers : int, optional
            Number of threads to use for training the model.
        epsilon : float, optional
            Constant used for clipping embeddings below a norm of one.
        regularization_coeff : float, optional
            Coefficient used for l2-regularization while training (0 effectively disables regularization).
        burn_in : int, optional
            Number of epochs to use for burn-in initialization (0 means no burn-in).
        burn_in_alpha : float, optional
            Learning rate for burn-in initialization, ignored if `burn_in` is 0.
        init_range : 2-tuple (float, float)
            Range within which the vectors are randomly initialized.
        dtype : numpy.dtype
            The numpy dtype to use for the vectors in the model (numpy.float64, numpy.float32 etc).
            Using lower precision floats may be useful in increasing training speed and reducing memory usage.
        seed : int, optional
            Seed for random to ensure reproducibility.

        Examples
        --------
        Initialize a model from a list:

        .. sourcecode:: pycon

            >>> from gensim.models.poincare import PoincareModel
            >>> relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat')]
            >>> model = PoincareModel(relations, negative=2)

        Initialize a model from a file containing one relation per line:

        .. sourcecode:: pycon

            >>> from gensim.models.poincare import PoincareModel, PoincareRelations
            >>> from gensim.test.utils import datapath
            >>> file_path = datapath('poincare_hypernyms.tsv')
            >>> model = PoincareModel(PoincareRelations(file_path), negative=2)

        See :class:`~gensim.models.poincare.PoincareRelations` for more options.

        """
        self.train_data = train_data
        self.kv = PoincareKeyedVectors(size, 0)
        self.all_relations = []
        self.node_relations = defaultdict(set)
        self._negatives_buffer = NegativesBuffer([])
        self._negatives_buffer_size = 2000
        self.size = size
        self.train_alpha = alpha
        self.burn_in_alpha = burn_in_alpha
        self.alpha = alpha
        self.negative = negative
        self.workers = workers
        self.epsilon = epsilon
        self.regularization_coeff = regularization_coeff
        self.burn_in = burn_in
        self._burn_in_done = False
        self.dtype = dtype
        self.seed = seed
        self._np_random = np_random.RandomState(seed)
        self.init_range = init_range
        self._loss_grad = None
        self.build_vocab(train_data)

    def build_vocab(self, relations, update=False):
        """Build the model's vocabulary from known relations.

        Parameters
        ----------
        relations : {iterable of (str, str), :class:`gensim.models.poincare.PoincareRelations`}
            Iterable of relations, e.g. a list of tuples, or a :class:`gensim.models.poincare.PoincareRelations`
            instance streaming from a file. Note that the relations are treated as ordered pairs,
            i.e. a relation (a, b) does not imply the opposite relation (b, a). In case the relations are symmetric,
            the data should contain both relations (a, b) and (b, a).
        update : bool, optional
            If true, only new nodes's embeddings are initialized.
            Use this when the model already has an existing vocabulary and you want to update it.
            If false, all node's embeddings are initialized.
            Use this when you're creating a new vocabulary from scratch.

        Examples
        --------
        Train a model and update vocab for online training:

        .. sourcecode:: pycon

            >>> from gensim.models.poincare import PoincareModel
            >>>
            >>> # train a new model from initial data
            >>> initial_relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal')]
            >>> model = PoincareModel(initial_relations, negative=1)
            >>> model.train(epochs=50)
            >>>
            >>> # online training: update the vocabulary and continue training
            >>> online_relations = [('striped_skunk', 'mammal')]
            >>> model.build_vocab(online_relations, update=True)
            >>> model.train(epochs=50)

        """
        old_index_to_key_len = len(self.kv.index_to_key)
        logger.info('loading relations from train data..')
        for relation in relations:
            if len(relation) != 2:
                raise ValueError('Relation pair "%s" should have exactly two items' % repr(relation))
            for item in relation:
                if item in self.kv.key_to_index:
                    self.kv.set_vecattr(item, 'count', self.kv.get_vecattr(item, 'count') + 1)
                else:
                    self.kv.key_to_index[item] = len(self.kv.index_to_key)
                    self.kv.index_to_key.append(item)
                    self.kv.set_vecattr(item, 'count', 1)
            node_1, node_2 = relation
            node_1_index, node_2_index = (self.kv.key_to_index[node_1], self.kv.key_to_index[node_2])
            self.node_relations[node_1_index].add(node_2_index)
            relation = (node_1_index, node_2_index)
            self.all_relations.append(relation)
        logger.info('loaded %d relations from train data, %d nodes', len(self.all_relations), len(self.kv))
        self.indices_set = set(range(len(self.kv.index_to_key)))
        self.indices_array = np.fromiter(range(len(self.kv.index_to_key)), dtype=int)
        self._init_node_probabilities()
        if not update:
            self._init_embeddings()
        else:
            self._update_embeddings(old_index_to_key_len)

    def _init_embeddings(self):
        """Randomly initialize vectors for the items in the vocab."""
        shape = (len(self.kv.index_to_key), self.size)
        self.kv.vectors = self._np_random.uniform(self.init_range[0], self.init_range[1], shape).astype(self.dtype)

    def _update_embeddings(self, old_index_to_key_len):
        """Randomly initialize vectors for the items in the additional vocab."""
        shape = (len(self.kv.index_to_key) - old_index_to_key_len, self.size)
        v = self._np_random.uniform(self.init_range[0], self.init_range[1], shape).astype(self.dtype)
        self.kv.vectors = np.concatenate([self.kv.vectors, v])

    def _init_node_probabilities(self):
        """Initialize a-priori probabilities."""
        counts = self.kv.expandos['count'].astype(np.float64)
        self._node_counts_cumsum = np.cumsum(counts)
        self._node_probabilities = counts / counts.sum()

    def _get_candidate_negatives(self):
        """Get candidate negatives of size `self.negative` from the negative examples buffer.

        Returns
        -------
        numpy.array
            Array of shape (`self.negative`,) containing indices of negative nodes.

        """
        if self._negatives_buffer.num_items() < self.negative:
            max_cumsum_value = self._node_counts_cumsum[-1]
            uniform_numbers = self._np_random.randint(1, max_cumsum_value + 1, self._negatives_buffer_size)
            cumsum_table_indices = np.searchsorted(self._node_counts_cumsum, uniform_numbers)
            self._negatives_buffer = NegativesBuffer(cumsum_table_indices)
        return self._negatives_buffer.get_items(self.negative)

    def _sample_negatives(self, node_index):
        """Get a sample of negatives for the given node.

        Parameters
        ----------
        node_index : int
            Index of the positive node for which negative samples are to be returned.

        Returns
        -------
        numpy.array
            Array of shape (self.negative,) containing indices of negative nodes for the given node index.

        """
        node_relations = self.node_relations[node_index]
        num_remaining_nodes = len(self.kv) - len(node_relations)
        if num_remaining_nodes < self.negative:
            raise ValueError('Cannot sample %d negative nodes from a set of %d negative nodes for %s' % (self.negative, num_remaining_nodes, self.kv.index_to_key[node_index]))
        positive_fraction = float(len(node_relations)) / len(self.kv)
        if positive_fraction < 0.01:
            indices = self._get_candidate_negatives()
            unique_indices = set(indices)
            times_sampled = 1
            while len(indices) != len(unique_indices) or unique_indices & node_relations:
                times_sampled += 1
                indices = self._get_candidate_negatives()
                unique_indices = set(indices)
            if times_sampled > 1:
                logger.debug('sampled %d times, positive fraction %.5f', times_sampled, positive_fraction)
        else:
            valid_negatives = np.array(list(self.indices_set - node_relations))
            probs = self._node_probabilities[valid_negatives]
            probs /= probs.sum()
            indices = self._np_random.choice(valid_negatives, size=self.negative, p=probs, replace=False)
        return list(indices)

    @staticmethod
    def _loss_fn(matrix, regularization_coeff=1.0):
        """Computes loss value.

        Parameters
        ----------
        matrix : numpy.array
            Array containing vectors for u, v and negative samples, of shape (2 + negative_size, dim).
        regularization_coeff : float, optional
            Coefficient to use for l2-regularization

        Returns
        -------
        float
            Computed loss value.

        Warnings
        --------
        Only used for autograd gradients, since autograd requires a specific function signature.

        """
        vector_u = matrix[0]
        vectors_v = matrix[1:]
        euclidean_dists = grad_np.linalg.norm(vector_u - vectors_v, axis=1)
        norm = grad_np.linalg.norm(vector_u)
        all_norms = grad_np.linalg.norm(vectors_v, axis=1)
        poincare_dists = grad_np.arccosh(1 + 2 * (euclidean_dists ** 2 / ((1 - norm ** 2) * (1 - all_norms ** 2))))
        exp_negative_distances = grad_np.exp(-poincare_dists)
        regularization_term = regularization_coeff * grad_np.linalg.norm(vectors_v[0]) ** 2
        return -grad_np.log(exp_negative_distances[0] / exp_negative_distances.sum()) + regularization_term

    @staticmethod
    def _clip_vectors(vectors, epsilon):
        """Clip vectors to have a norm of less than one.

        Parameters
        ----------
        vectors : numpy.array
            Can be 1-D, or 2-D (in which case the norm for each row is checked).
        epsilon : float
            Parameter for numerical stability, each dimension of the vector is reduced by `epsilon`
            if the norm of the vector is greater than or equal to 1.

        Returns
        -------
        numpy.array
            Array with norms clipped below 1.

        """
        one_d = len(vectors.shape) == 1
        threshold = 1 - epsilon
        if one_d:
            norm = np.linalg.norm(vectors)
            if norm < threshold:
                return vectors
            else:
                return vectors / norm - np.sign(vectors) * epsilon
        else:
            norms = np.linalg.norm(vectors, axis=1)
            if (norms < threshold).all():
                return vectors
            else:
                vectors[norms >= threshold] *= (threshold / norms[norms >= threshold])[:, np.newaxis]
                vectors[norms >= threshold] -= np.sign(vectors[norms >= threshold]) * epsilon
                return vectors

    def save(self, *args, **kwargs):
        """Save complete model to disk, inherited from :class:`~gensim.utils.SaveLoad`.

        See also
        --------
        :meth:`~gensim.models.poincare.PoincareModel.load`

        Parameters
        ----------
        *args
            Positional arguments passed to :meth:`~gensim.utils.SaveLoad.save`.
        **kwargs
            Keyword arguments passed to :meth:`~gensim.utils.SaveLoad.save`.

        """
        self._loss_grad = None
        attrs_to_ignore = ['_node_probabilities', '_node_counts_cumsum']
        kwargs['ignore'] = set(list(kwargs.get('ignore', [])) + attrs_to_ignore)
        super(PoincareModel, self).save(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        """Load model from disk, inherited from :class:`~gensim.utils.SaveLoad`.

        See also
        --------
        :meth:`~gensim.models.poincare.PoincareModel.save`

        Parameters
        ----------
        *args
            Positional arguments passed to :meth:`~gensim.utils.SaveLoad.load`.
        **kwargs
            Keyword arguments passed to :meth:`~gensim.utils.SaveLoad.load`.

        Returns
        -------
        :class:`~gensim.models.poincare.PoincareModel`
            The loaded model.

        """
        model = super(PoincareModel, cls).load(*args, **kwargs)
        model._init_node_probabilities()
        return model

    def _prepare_training_batch(self, relations, all_negatives, check_gradients=False):
        """Create a training batch and compute gradients and loss for the batch.

        Parameters
        ----------
        relations : list of tuples
            List of tuples of positive examples of the form (node_1_index, node_2_index).
        all_negatives : list of lists
            List of lists of negative samples for each node_1 in the positive examples.
        check_gradients : bool, optional
            Whether to compare the computed gradients to autograd gradients for this batch.

        Returns
        -------
        :class:`~gensim.models.poincare.PoincareBatch`
            Node indices, computed gradients and loss for the batch.

        """
        batch_size = len(relations)
        indices_u, indices_v = ([], [])
        for relation, negatives in zip(relations, all_negatives):
            u, v = relation
            indices_u.append(u)
            indices_v.append(v)
            indices_v.extend(negatives)
        vectors_u = self.kv.vectors[indices_u]
        vectors_v = self.kv.vectors[indices_v].reshape((batch_size, 1 + self.negative, self.size))
        vectors_v = vectors_v.swapaxes(0, 1).swapaxes(1, 2)
        batch = PoincareBatch(vectors_u, vectors_v, indices_u, indices_v, self.regularization_coeff)
        batch.compute_all()
        if check_gradients:
            self._check_gradients(relations, all_negatives, batch)
        return batch

    def _check_gradients(self, relations, all_negatives, batch, tol=1e-08):
        """Compare computed gradients for batch to autograd gradients.

        Parameters
        ----------
        relations : list of tuples
            List of tuples of positive examples of the form (node_1_index, node_2_index).
        all_negatives : list of lists
            List of lists of negative samples for each node_1 in the positive examples.
        batch : :class:`~gensim.models.poincare.PoincareBatch`
            Batch for which computed gradients are to be checked.
        tol : float, optional
            The maximum error between our computed gradients and the reference ones from autograd.

        """
        if not AUTOGRAD_PRESENT:
            logger.warning('autograd could not be imported, cannot do gradient checking')
            logger.warning('please install autograd to enable gradient checking')
            return
        if self._loss_grad is None:
            self._loss_grad = grad(PoincareModel._loss_fn)
        max_diff = 0.0
        for i, (relation, negatives) in enumerate(zip(relations, all_negatives)):
            u, v = relation
            auto_gradients = self._loss_grad(np.vstack((self.kv.vectors[u], self.kv.vectors[[v] + negatives])), self.regularization_coeff)
            computed_gradients = np.vstack((batch.gradients_u[:, i], batch.gradients_v[:, :, i]))
            diff = np.abs(auto_gradients - computed_gradients).max()
            if diff > max_diff:
                max_diff = diff
        logger.info('max difference between computed gradients and autograd gradients: %.10f', max_diff)
        assert max_diff < tol, 'Max difference between computed gradients and autograd gradients %.10f, greater than tolerance %.10f' % (max_diff, tol)

    def _sample_negatives_batch(self, nodes):
        """Get negative examples for each node.

        Parameters
        ----------
        nodes : iterable of int
            Iterable of node indices for which negative samples are to be returned.

        Returns
        -------
        list of lists
            Each inner list is a list of negative samples for a single node in the input list.

        """
        all_indices = [self._sample_negatives(node) for node in nodes]
        return all_indices

    def _train_on_batch(self, relations, check_gradients=False):
        """Perform training for a single training batch.

        Parameters
        ----------
        relations : list of tuples of (int, int)
            List of tuples of positive examples of the form (node_1_index, node_2_index).
        check_gradients : bool, optional
            Whether to compare the computed gradients to autograd gradients for this batch.

        Returns
        -------
        :class:`~gensim.models.poincare.PoincareBatch`
            The batch that was just trained on, contains computed loss for the batch.

        """
        all_negatives = self._sample_negatives_batch((relation[0] for relation in relations))
        batch = self._prepare_training_batch(relations, all_negatives, check_gradients)
        self._update_vectors_batch(batch)
        return batch

    @staticmethod
    def _handle_duplicates(vector_updates, node_indices):
        """Handle occurrences of multiple updates to the same node in a batch of vector updates.

        Parameters
        ----------
        vector_updates : numpy.array
            Array with each row containing updates to be performed on a certain node.
        node_indices : list of int
            Node indices on which the above updates are to be performed on.

        Notes
        -----
        Mutates the `vector_updates` array.

        Required because vectors[[2, 1, 2]] += np.array([-0.5, 1.0, 0.5]) performs only the last update
        on the row at index 2.

        """
        counts = Counter(node_indices)
        node_dict = defaultdict(list)
        for i, node_index in enumerate(node_indices):
            node_dict[node_index].append(i)
        for node_index, count in counts.items():
            if count == 1:
                continue
            positions = node_dict[node_index]
            vector_updates[positions[-1]] = vector_updates[positions].sum(axis=0)
            vector_updates[positions[:-1]] = 0

    def _update_vectors_batch(self, batch):
        """Update vectors for nodes in the given batch.

        Parameters
        ----------
        batch : :class:`~gensim.models.poincare.PoincareBatch`
            Batch containing computed gradients and node indices of the batch for which updates are to be done.

        """
        grad_u, grad_v = (batch.gradients_u, batch.gradients_v)
        indices_u, indices_v = (batch.indices_u, batch.indices_v)
        batch_size = len(indices_u)
        u_updates = (self.alpha * batch.alpha ** 2 / 4 * grad_u).T
        self._handle_duplicates(u_updates, indices_u)
        self.kv.vectors[indices_u] -= u_updates
        self.kv.vectors[indices_u] = self._clip_vectors(self.kv.vectors[indices_u], self.epsilon)
        v_updates = self.alpha * (batch.beta ** 2)[:, np.newaxis] / 4 * grad_v
        v_updates = v_updates.swapaxes(1, 2).swapaxes(0, 1)
        v_updates = v_updates.reshape(((1 + self.negative) * batch_size, self.size))
        self._handle_duplicates(v_updates, indices_v)
        self.kv.vectors[indices_v] -= v_updates
        self.kv.vectors[indices_v] = self._clip_vectors(self.kv.vectors[indices_v], self.epsilon)

    def train(self, epochs, batch_size=10, print_every=1000, check_gradients_every=None):
        """Train Poincare embeddings using loaded data and model parameters.

        Parameters
        ----------
        epochs : int
            Number of iterations (epochs) over the corpus.
        batch_size : int, optional
            Number of examples to train on in a single batch.

        print_every : int, optional
            Prints progress and average loss after every `print_every` batches.
        check_gradients_every : int or None, optional
            Compares computed gradients and autograd gradients after every `check_gradients_every` batches.
            Useful for debugging, doesn't compare by default.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.models.poincare import PoincareModel
            >>> relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat')]
            >>> model = PoincareModel(relations, negative=2)
            >>> model.train(epochs=50)

        """
        if self.workers > 1:
            raise NotImplementedError('Multi-threaded version not implemented yet')
        old_settings = np.seterr(divide='ignore', invalid='ignore')
        logger.info('training model of size %d with %d workers on %d relations for %d epochs and %d burn-in epochs, using lr=%.5f burn-in lr=%.5f negative=%d', self.size, self.workers, len(self.all_relations), epochs, self.burn_in, self.alpha, self.burn_in_alpha, self.negative)
        if self.burn_in > 0 and (not self._burn_in_done):
            logger.info('starting burn-in (%d epochs)----------------------------------------', self.burn_in)
            self.alpha = self.burn_in_alpha
            self._train_batchwise(epochs=self.burn_in, batch_size=batch_size, print_every=print_every, check_gradients_every=check_gradients_every)
            self._burn_in_done = True
            logger.info('burn-in finished')
        self.alpha = self.train_alpha
        logger.info('starting training (%d epochs)----------------------------------------', epochs)
        self._train_batchwise(epochs=epochs, batch_size=batch_size, print_every=print_every, check_gradients_every=check_gradients_every)
        logger.info('training finished')
        np.seterr(**old_settings)

    def _train_batchwise(self, epochs, batch_size=10, print_every=1000, check_gradients_every=None):
        """Train Poincare embeddings using specified parameters.

        Parameters
        ----------
        epochs : int
            Number of iterations (epochs) over the corpus.
        batch_size : int, optional
            Number of examples to train on in a single batch.
        print_every : int, optional
            Prints progress and average loss after every `print_every` batches.
        check_gradients_every : int or None, optional
            Compares computed gradients and autograd gradients after every `check_gradients_every` batches.
            Useful for debugging, doesn't compare by default.

        """
        if self.workers > 1:
            raise NotImplementedError('Multi-threaded version not implemented yet')
        for epoch in range(1, epochs + 1):
            indices = list(range(len(self.all_relations)))
            self._np_random.shuffle(indices)
            avg_loss = 0.0
            last_time = time.time()
            for batch_num, i in enumerate(range(0, len(indices), batch_size), start=1):
                should_print = not batch_num % print_every
                check_gradients = bool(check_gradients_every) and batch_num % check_gradients_every == 0
                batch_indices = indices[i:i + batch_size]
                relations = [self.all_relations[idx] for idx in batch_indices]
                result = self._train_on_batch(relations, check_gradients=check_gradients)
                avg_loss += result.loss
                if should_print:
                    avg_loss /= print_every
                    time_taken = time.time() - last_time
                    speed = print_every * batch_size / time_taken
                    logger.info('training on epoch %d, examples #%d-#%d, loss: %.2f' % (epoch, i, i + batch_size, avg_loss))
                    logger.info('time taken for %d examples: %.2f s, %.2f examples / s' % (print_every * batch_size, time_taken, speed))
                    last_time = time.time()
                    avg_loss = 0.0