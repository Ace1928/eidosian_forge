import gensim
import logging
import copy
import sys
import numpy as np
class CallbackAny2Vec:
    """Base class to build callbacks for :class:`~gensim.models.word2vec.Word2Vec` & subclasses.

    Callbacks are used to apply custom functions over the model at specific points
    during training (epoch start, batch end etc.). This is a base class and its purpose is to be inherited by
    custom Callbacks that implement one or more of its methods (depending on the point during training where they
    want some action to be taken).

    See examples at the module level docstring for how to define your own callbacks by inheriting  from this class.

    As of gensim 4.0.0, the following callbacks are no longer supported, and overriding them will have no effect:

    - on_batch_begin
    - on_batch_end

    """

    def on_epoch_begin(self, model):
        """Method called at the start of each epoch.

        Parameters
        ----------
        model : :class:`~gensim.models.word2vec.Word2Vec` or subclass
            Current model.

        """
        pass

    def on_epoch_end(self, model):
        """Method called at the end of each epoch.

        Parameters
        ----------
        model : :class:`~gensim.models.word2vec.Word2Vec` or subclass
            Current model.

        """
        pass

    def on_train_begin(self, model):
        """Method called at the start of the training process.

        Parameters
        ----------
        model : :class:`~gensim.models.word2vec.Word2Vec` or subclass
            Current model.

        """
        pass

    def on_train_end(self, model):
        """Method called at the end of the training process.

        Parameters
        ----------
        model : :class:`~gensim.models.word2vec.Word2Vec` or subclass
            Current model.

        """
        pass