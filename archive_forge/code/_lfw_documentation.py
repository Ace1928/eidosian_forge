import logging
from numbers import Integral, Real
from os import PathLike, listdir, makedirs, remove
from os.path import exists, isdir, join
import numpy as np
from joblib import Memory
from ..utils import Bunch
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ._base import (
Load the Labeled Faces in the Wild (LFW) pairs dataset (classification).

    Download it if necessary.

    =================   =======================
    Classes                                   2
    Samples total                         13233
    Dimensionality                         5828
    Features            real, between 0 and 255
    =================   =======================

    In the official `README.txt`_ this task is described as the
    "Restricted" task.  As I am not sure as to implement the
    "Unrestricted" variant correctly, I left it as unsupported for now.

      .. _`README.txt`: http://vis-www.cs.umass.edu/lfw/README.txt

    The original images are 250 x 250 pixels, but the default slice and resize
    arguments reduce them to 62 x 47.

    Read more in the :ref:`User Guide <labeled_faces_in_the_wild_dataset>`.

    Parameters
    ----------
    subset : {'train', 'test', '10_folds'}, default='train'
        Select the dataset to load: 'train' for the development training
        set, 'test' for the development test set, and '10_folds' for the
        official evaluation set that is meant to be used with a 10-folds
        cross validation.

    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By
        default all scikit-learn data is stored in '~/scikit_learn_data'
        subfolders.

    funneled : bool, default=True
        Download and use the funneled variant of the dataset.

    resize : float, default=0.5
        Ratio used to resize the each face picture.

    color : bool, default=False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than the shape with color = False.

    slice_ : tuple of slice, default=(slice(70, 195), slice(78, 172))
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray of shape (2200, 5828). Shape depends on ``subset``.
            Each row corresponds to 2 ravel'd face images
            of original size 62 x 47 pixels.
            Changing the ``slice_``, ``resize`` or ``subset`` parameters
            will change the shape of the output.
        pairs : ndarray of shape (2200, 2, 62, 47). Shape depends on ``subset``
            Each row has 2 face images corresponding
            to same or different person from the dataset
            containing 5749 people. Changing the ``slice_``,
            ``resize`` or ``subset`` parameters will change the shape of the
            output.
        target : numpy array of shape (2200,). Shape depends on ``subset``.
            Labels associated to each pair of images.
            The two label values being different persons or the same person.
        target_names : numpy array of shape (2,)
            Explains the target values of the target array.
            0 corresponds to "Different person", 1 corresponds to "same person".
        DESCR : str
            Description of the Labeled Faces in the Wild (LFW) dataset.
    