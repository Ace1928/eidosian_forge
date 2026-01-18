from contextlib import contextmanager  # noqa E402
from copy import deepcopy
import logging
import sys
import os
from collections import OrderedDict, defaultdict
from six import iteritems, string_types, integer_types
import warnings
import numpy as np
import ctypes
import platform
import tempfile
import shutil
import json
from enum import Enum
from operator import itemgetter
import threading
import scipy.sparse
from .plot_helpers import save_plot_file, try_plot_offline, OfflineMetricVisualizer
from . import _catboost
from .metrics import BuiltinMetric
class CatBoostClassifier(CatBoost):
    """
    Implementation of the scikit-learn API for CatBoost classification.

    Parameters
    ----------
    iterations : int, [default=500]
        Max count of trees.
        range: [1,+inf)
    learning_rate : float, [default value is selected automatically for binary classification with other parameters set to default. In all other cases default is 0.03]
        Step size shrinkage used in update to prevents overfitting.
        range: (0,1]
    depth : int, [default=6]
        Depth of a tree. All trees are the same depth.
        range: [1,16]
    l2_leaf_reg : float, [default=3.0]
        Coefficient at the L2 regularization term of the cost function.
        range: [0,+inf)
    model_size_reg : float, [default=None]
        Model size regularization coefficient.
        range: [0,+inf)
    rsm : float, [default=None]
        Subsample ratio of columns when constructing each tree.
        range: (0,1]
    loss_function : string or object, [default='Logloss']
        The metric to use in training and also selector of the machine learning
        problem to solve. If string, then the name of a supported metric,
        optionally suffixed with parameter description.
        If object, it shall provide methods 'calc_ders_range' or 'calc_ders_multi'.
    border_count : int, [default = 254 for training on CPU or 128 for training on GPU]
        The number of partitions in numeric features binarization. Used in the preliminary calculation.
        range: [1,65535] on CPU, [1,255] on GPU
    feature_border_type : string, [default='GreedyLogSum']
        The binarization mode in numeric features binarization. Used in the preliminary calculation.
        Possible values:
            - 'Median'
            - 'Uniform'
            - 'UniformAndQuantiles'
            - 'GreedyLogSum'
            - 'MaxLogSum'
            - 'MinEntropy'
    per_float_feature_quantization : list of strings, [default=None]
        List of float binarization descriptions.
        Format : described in documentation on catboost.ai
        Example 1: ['0:1024'] means that feature 0 will have 1024 borders.
        Example 2: ['0:border_count=1024', '1:border_count=1024', ...] means that two first features have 1024 borders.
        Example 3: ['0:nan_mode=Forbidden,border_count=32,border_type=GreedyLogSum',
                    '1:nan_mode=Forbidden,border_count=32,border_type=GreedyLogSum'] - defines more quantization properties for first two features.
    input_borders : string or pathlib.Path, [default=None]
        input file with borders used in numeric features binarization.
    output_borders : string, [default=None]
        output file for borders that were used in numeric features binarization.
    fold_permutation_block : int, [default=1]
        To accelerate the learning.
        The recommended value is within [1, 256]. On small samples, must be set to 1.
        range: [1,+inf)
    od_pval : float, [default=None]
        Use overfitting detector to stop training when reaching a specified threshold.
        Can be used only with eval_set.
        range: [0,1]
    od_wait : int, [default=None]
        Number of iterations which overfitting detector will wait after new best error.
    od_type : string, [default=None]
        Type of overfitting detector which will be used in program.
        Posible values:
            - 'IncToDec'
            - 'Iter'
        For 'Iter' type od_pval must not be set.
        If None, then od_type=IncToDec.
    nan_mode : string, [default=None]
        Way to process missing values for numeric features.
        Possible values:
            - 'Forbidden' - raises an exception if there is a missing value for a numeric feature in a dataset.
            - 'Min' - each missing value will be processed as the minimum numerical value.
            - 'Max' - each missing value will be processed as the maximum numerical value.
        If None, then nan_mode=Min.
    counter_calc_method : string, [default=None]
        The method used to calculate counters for dataset with Counter type.
        Possible values:
            - 'PrefixTest' - only objects up to current in the test dataset are considered
            - 'FullTest' - all objects are considered in the test dataset
            - 'SkipTest' - Objects from test dataset are not considered
            - 'Full' - all objects are considered for both learn and test dataset
        If None, then counter_calc_method=PrefixTest.
    leaf_estimation_iterations : int, [default=None]
        The number of steps in the gradient when calculating the values in the leaves.
        If None, then leaf_estimation_iterations=1.
        range: [1,+inf)
    leaf_estimation_method : string, [default=None]
        The method used to calculate the values in the leaves.
        Possible values:
            - 'Newton'
            - 'Gradient'
    thread_count : int, [default=None]
        Number of parallel threads used to run CatBoost.
        If None or -1, then the number of threads is set to the number of CPU cores.
        range: [1,+inf)
    random_seed : int, [default=None]
        Random number seed.
        If None, 0 is used.
        range: [0,+inf)
    use_best_model : bool, [default=None]
        To limit the number of trees in predict() using information about the optimal value of the error function.
        Can be used only with eval_set.
    best_model_min_trees : int, [default=None]
        The minimal number of trees the best model should have.
    verbose: bool
        When set to True, logging_level is set to 'Verbose'.
        When set to False, logging_level is set to 'Silent'.
    silent: bool, synonym for verbose
    logging_level : string, [default='Verbose']
        Possible values:
            - 'Silent'
            - 'Verbose'
            - 'Info'
            - 'Debug'
    metric_period : int, [default=1]
        The frequency of iterations to print the information to stdout. The value should be a positive integer.
    simple_ctr: list of strings, [default=None]
        Binarization settings for categorical features.
            Format : see documentation
            Example: ['Borders:CtrBorderCount=5:Prior=0:Prior=0.5', 'BinarizedTargetMeanValue:TargetBorderCount=10:TargetBorderType=MinEntropy', ...]
            CTR types:
                CPU and GPU
                - 'Borders'
                - 'Buckets'
                CPU only
                - 'BinarizedTargetMeanValue'
                - 'Counter'
                GPU only
                - 'FloatTargetMeanValue'
                - 'FeatureFreq'
            Number_of_borders, binarization type, target borders and binarizations, priors are optional parametrs
    combinations_ctr: list of strings, [default=None]
    per_feature_ctr: list of strings, [default=None]
    ctr_target_border_count: int, [default=None]
        Maximum number of borders used in target binarization for categorical features that need it.
        If TargetBorderCount is specified in 'simple_ctr', 'combinations_ctr' or 'per_feature_ctr' option it
        overrides this value.
        range: [1, 255]
    ctr_leaf_count_limit : int, [default=None]
        The maximum number of leaves with categorical features.
        If the number of leaves exceeds the specified limit, some leaves are discarded.
        The leaves to be discarded are selected as follows:
            - The leaves are sorted by the frequency of the values.
            - The top N leaves are selected, where N is the value specified in the parameter.
            - All leaves starting from N+1 are discarded.
        This option reduces the resulting model size
        and the amount of memory required for training.
        Note that the resulting quality of the model can be affected.
        range: [1,+inf) (for zero limit use ignored_features)
    store_all_simple_ctr : bool, [default=None]
        Ignore categorical features, which are not used in feature combinations,
        when choosing candidates for exclusion.
        Use this parameter with ctr_leaf_count_limit only.
    max_ctr_complexity : int, [default=4]
        The maximum number of Categ features that can be combined.
        range: [0,+inf)
    has_time : bool, [default=False]
        To use the order in which objects are represented in the input data
        (do not perform a random permutation of the dataset at the preprocessing stage).
    allow_const_label : bool, [default=False]
        To allow the constant label value in dataset.
    target_border: float, [default=None]
        Border for target binarization.
    classes_count : int, [default=None]
        The upper limit for the numeric class label.
        Defines the number of classes for multiclassification.
        Only non-negative integers can be specified.
        The given integer should be greater than any of the target values.
        If this parameter is specified the labels for all classes in the input dataset
        should be smaller than the given value.
        If several of 'classes_count', 'class_weights', 'class_names' parameters are defined
        the numbers of classes specified by each of them must be equal.
    class_weights : list or dict, [default=None]
        Classes weights. The values are used as multipliers for the object weights.
        If None, all classes are supposed to have weight one.
        If list - class weights in order of class_names or sequential classes if class_names is undefined
        If dict - dict of class_name -> class_weight.
        If several of 'classes_count', 'class_weights', 'class_names' parameters are defined
        the numbers of classes specified by each of them must be equal.
    auto_class_weights : string [default=None]
        Enables automatic class weights calculation. Possible values:
            - Balanced  # weight = maxSummaryClassWeight / summaryClassWeight, statistics determined from train pool
            - SqrtBalanced  # weight = sqrt(maxSummaryClassWeight / summaryClassWeight)
    class_names: list of strings, [default=None]
        Class names. Allows to redefine the default values for class labels (integer numbers).
        If several of 'classes_count', 'class_weights', 'class_names' parameters are defined
        the numbers of classes specified by each of them must be equal.
    one_hot_max_size : int, [default=None]
        Convert the feature to float
        if the number of different values that it takes exceeds the specified value.
        Ctrs are not calculated for such features.
    random_strength : float, [default=1]
        Score standard deviation multiplier.
    random_score_type : string [default=None]
        Type of random noise added to scores.
        Possible values:
            - 'Gumbel' - Gumbel-distributed
            - 'NormalWithModelSizeDecrease' - Normally-distributed with deviation decreasing with model iteration count
        If None than 'NormalWithModelSizeDecrease' will be used by default.
    name : string, [default='experiment']
        The name that should be displayed in the visualization tools.
    ignored_features : list, [default=None]
        Indices or names of features that should be excluded when training.
    train_dir : string or pathlib.Path, [default=None]
        The directory in which you want to record generated in the process of learning files.
    custom_metric : string or list of strings, [default=None]
        To use your own metric function.
    custom_loss: alias to custom_metric
    eval_metric : string or object, [default=None]
        To optimize your custom metric in loss.
    bagging_temperature : float, [default=None]
        Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is.
        Typical values are in range [0, 1] (0 - no bagging, 1 - default).
    save_snapshot : bool, [default=None]
        Enable progress snapshotting for restoring progress after crashes or interruptions
    snapshot_file : string or pathlib.Path, [default=None]
        Learn progress snapshot file path, if None will use default filename
    snapshot_interval: int, [default=600]
        Interval between saving snapshots (seconds)
    fold_len_multiplier : float, [default=None]
        Fold length multiplier. Should be greater than 1
    used_ram_limit : string or number, [default=None]
        Set a limit on memory consumption (value like '1.2gb' or 1.2e9).
        WARNING: Currently this option affects CTR memory usage only.
    gpu_ram_part : float, [default=0.95]
        Fraction of the GPU RAM to use for training, a value from (0, 1].
    pinned_memory_size: int [default=None]
        Size of additional CPU pinned memory used for GPU learning,
        usually is estimated automatically, thus usually should not be set.
    allow_writing_files : bool, [default=True]
        If this flag is set to False, no files with different diagnostic info will be created during training.
        With this flag no snapshotting can be done. Plus visualisation will not
        work, because visualisation uses files that are created and updated during training.
    final_ctr_computation_mode : string, [default='Default']
        Possible values:
            - 'Default' - Compute final ctrs for all pools.
            - 'Skip' - Skip final ctr computation. WARNING: model without ctrs can't be applied.
    approx_on_full_history : bool, [default=False]
        If this flag is set to True, each approximated value is calculated using all the preceeding rows in the fold (slower, more accurate).
        If this flag is set to False, each approximated value is calculated using only the beginning 1/fold_len_multiplier fraction of the fold (faster, slightly less accurate).
    boosting_type : string, default value depends on object count and feature count in train dataset and on learning mode.
        Boosting scheme.
        Possible values:
            - 'Ordered' - Gives better quality, but may slow down the training.
            - 'Plain' - The classic gradient boosting scheme. May result in quality degradation, but does not slow down the training.
    task_type : string, [default=None]
        The calcer type used to train the model.
        Possible values:
            - 'CPU'
            - 'GPU'
    device_config : string, [default=None], deprecated, use devices instead
    devices : list or string, [default=None], GPU devices to use.
        String format is: '0' for 1 device or '0:1:3' for multiple devices or '0-3' for range of devices.
        List format is : [0] for 1 device or [0,1,3] for multiple devices.

    bootstrap_type : string, Bayesian, Bernoulli, Poisson, MVS.
        Default bootstrap is Bayesian for GPU and MVS for CPU.
        Poisson bootstrap is supported only on GPU.
        MVS bootstrap is supported only on GPU.

    subsample : float, [default=None]
        Sample rate for bagging. This parameter can be used Poisson or Bernoully bootstrap types.

    mvs_reg : float, [default is set automatically at each iteration based on gradient distribution]
        Regularization parameter for MVS sampling algorithm

    monotone_constraints : list or numpy.ndarray or string or dict, [default=None]
        Monotone constraints for features.

    feature_weights : list or numpy.ndarray or string or dict, [default=None]
        Coefficient to multiply split gain with specific feature use. Should be non-negative.

    penalties_coefficient : float, [default=1]
        Common coefficient for all penalties. Should be non-negative.

    first_feature_use_penalties : list or numpy.ndarray or string or dict, [default=None]
        Penalties to first use of specific feature in model. Should be non-negative.

    per_object_feature_penalties : list or numpy.ndarray or string or dict, [default=None]
        Penalties for first use of feature for each object. Should be non-negative.

    sampling_frequency : string, [default=PerTree]
        Frequency to sample weights and objects when building trees.
        Possible values:
            - 'PerTree' - Before constructing each new tree
            - 'PerTreeLevel' - Before choosing each new split of a tree

    sampling_unit : string, [default='Object'].
        Possible values:
            - 'Object'
            - 'Group'
        The parameter allows to specify the sampling scheme:
        sample weights for each object individually or for an entire group of objects together.

    dev_score_calc_obj_block_size: int, [default=5000000]
        CPU only. Size of block of samples in score calculation. Should be > 0
        Used only for learning speed tuning.
        Changing this parameter can affect results due to numerical accuracy differences

    dev_efb_max_buckets : int, [default=1024]
        CPU only. Maximum bucket count in exclusive features bundle. Should be in an integer between 0 and 65536.
        Used only for learning speed tuning.

    sparse_features_conflict_fraction : float, [default=0.0]
        CPU only. Maximum allowed fraction of conflicting non-default values for features in exclusive features bundle.
        Should be a real value in [0, 1) interval.

    grow_policy : string, [SymmetricTree,Lossguide,Depthwise], [default=SymmetricTree]
        The tree growing policy. It describes how to perform greedy tree construction.

    min_data_in_leaf : int, [default=1].
        The minimum training samples count in leaf.
        CatBoost will not search for new splits in leaves with samples count less than min_data_in_leaf.
        This parameter is used only for Depthwise and Lossguide growing policies.

    max_leaves : int, [default=31],
        The maximum leaf count in resulting tree.
        This parameter is used only for Lossguide growing policy.

    score_function : string, possible values L2, Cosine, NewtonL2, NewtonCosine, [default=Cosine]
        For growing policy Lossguide default=NewtonL2.
        GPU only. Score that is used during tree construction to select the next tree split.

    max_depth : int, Synonym for depth.

    n_estimators : int, synonym for iterations.

    num_trees : int, synonym for iterations.

    num_boost_round : int, synonym for iterations.

    colsample_bylevel : float, synonym for rsm.

    random_state : int, synonym for random_seed.

    reg_lambda : float, synonym for l2_leaf_reg.

    objective : string, synonym for loss_function.

    num_leaves : int, synonym for max_leaves.

    min_child_samples : int, synonym for min_data_in_leaf

    eta : float, synonym for learning_rate.

    max_bin : float, synonym for border_count.

    scale_pos_weight : float, synonym for class_weights.
        Can be used only for binary classification. Sets weight multiplier for
        class 1 to scale_pos_weight value.

    metadata : dict, string to string key-value pairs to be stored in model metadata storage

    early_stopping_rounds : int
        Synonym for od_wait. Only one of these parameters should be set.

    cat_features : list or numpy.ndarray, [default=None]
        If not None, giving the list of Categ features indices or names (names are represented as strings).
        If it contains feature names, feature names must be defined for the training dataset passed to 'fit'.

    text_features : list or numpy.ndarray, [default=None]
        If not None, giving the list of Text features indices or names (names are represented as strings).
        If it contains feature names, feature names must be defined for the training dataset passed to 'fit'.

    embedding_features : list or numpy.ndarray, [default=None]
        If not None, giving the list of Embedding features indices or names (names are represented as strings).
        If it contains feature names, feature names must be defined for the training dataset passed to 'fit'.

    leaf_estimation_backtracking : string, [default=None]
        Type of backtracking during gradient descent.
        Possible values:
            - 'No' - never backtrack; supported on CPU and GPU
            - 'AnyImprovement' - reduce the descent step until the value of loss function is less than before the step; supported on CPU and GPU
            - 'Armijo' - reduce the descent step until Armijo condition is satisfied; supported on GPU only

    model_shrink_rate : float, [default=0]
        This parameter enables shrinkage of model at the start of each iteration. CPU only.
        For Constant mode shrinkage coefficient is calculated as (1 - model_shrink_rate * learning_rate).
        For Decreasing mode shrinkage coefficient is calculated as (1 - model_shrink_rate / iteration).
        Shrinkage coefficient should be in [0, 1).

    model_shrink_mode : string, [default=None]
        Mode of shrinkage coefficient calculation. CPU only.
        Possible values:
            - 'Constant' - Shrinkage coefficient is constant at each iteration.
            - 'Decreasing' - Shrinkage coefficient decreases at each iteration.

    langevin : bool, [default=False]
        Enables the Stochastic Gradient Langevin Boosting. CPU only.

    diffusion_temperature : float, [default=0]
        Langevin boosting diffusion temperature. CPU only.

    posterior_sampling : bool, [default=False]
        Set group of parameters for further use Uncertainty prediction:
            - Langevin = True
            - Model Shrink Rate = 1/(2N), where N is dataset size
            - Model Shrink Mode = Constant
            - Diffusion-temperature = N, where N is dataset size. CPU only.

    boost_from_average : bool, [default=True for RMSE, False for other losses]
        Enables to initialize approx values by best constant value for specified loss function.
        Available for RMSE, Logloss, CrossEntropy, Quantile and MAE.

    tokenizers : list of dicts,
        Each dict is a tokenizer description. Example:
        ```
        [
            {
                'tokenizer_id': 'Tokenizer',  # Tokeinzer identifier.
                'lowercasing': 'false',  # Possible values: 'true', 'false'.
                'number_process_policy': 'LeaveAsIs',  # Possible values: 'Skip', 'LeaveAsIs', 'Replace'.
                'number_token': '%',  # Rarely used character. Used in conjunction with Replace NumberProcessPolicy.
                'separator_type': 'ByDelimiter',  # Possible values: 'ByDelimiter', 'BySense'.
                'delimiter': ' ',  # Used in conjunction with ByDelimiter SeparatorType.
                'split_by_set': 'false',  # Each single character in delimiter used as individual delimiter.
                'skip_empty': 'true',  # Possible values: 'true', 'false'.
                'token_types': ['Word', 'Number', 'Unknown'],  # Used in conjunction with BySense SeparatorType.
                    # Possible values: 'Word', 'Number', 'Punctuation', 'SentenceBreak', 'ParagraphBreak', 'Unknown'.
                'subtokens_policy': 'SingleToken',  # Possible values:
                    # 'SingleToken' - All subtokens are interpreted as single token).
                    # 'SeveralTokens' - All subtokens are interpreted as several token.
            },
            ...
        ]
        ```

    dictionaries : list of dicts,
        Each dict is a tokenizer description. Example:
        ```
        [
            {
                'dictionary_id': 'Dictionary',  # Dictionary identifier.
                'token_level_type': 'Word',  # Possible values: 'Word', 'Letter'.
                'gram_order': '1',  # 1 for Unigram, 2 for Bigram, ...
                'skip_step': '0',  # 1 for 1-skip-gram, ...
                'end_of_word_token_policy': 'Insert',  # Possible values: 'Insert', 'Skip'.
                'end_of_sentence_token_policy': 'Skip',  # Possible values: 'Insert', 'Skip'.
                'occurrence_lower_bound': '3',  # The lower bound of token occurrences in the text to include it in the dictionary.
                'max_dictionary_size': '50000',  # The max dictionary size.
            },
            ...
        ]
        ```

    feature_calcers : list of strings,
        Each string is a calcer description. Example:
        ```
        [
            'NaiveBayes',
            'BM25',
            'BoW:top_tokens_count=2000',
        ]
        ```

    text_processing : dict,
        Text processging description.

    eval_fraction : float, [default=None]
        Fraction of the train dataset to be used as the evaluation dataset.
    """
    _estimator_type = 'classifier'

    def __init__(self, iterations=None, learning_rate=None, depth=None, l2_leaf_reg=None, model_size_reg=None, rsm=None, loss_function=None, border_count=None, feature_border_type=None, per_float_feature_quantization=None, input_borders=None, output_borders=None, fold_permutation_block=None, od_pval=None, od_wait=None, od_type=None, nan_mode=None, counter_calc_method=None, leaf_estimation_iterations=None, leaf_estimation_method=None, thread_count=None, random_seed=None, use_best_model=None, best_model_min_trees=None, verbose=None, silent=None, logging_level=None, metric_period=None, ctr_leaf_count_limit=None, store_all_simple_ctr=None, max_ctr_complexity=None, has_time=None, allow_const_label=None, target_border=None, classes_count=None, class_weights=None, auto_class_weights=None, class_names=None, one_hot_max_size=None, random_strength=None, random_score_type=None, name=None, ignored_features=None, train_dir=None, custom_loss=None, custom_metric=None, eval_metric=None, bagging_temperature=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None, fold_len_multiplier=None, used_ram_limit=None, gpu_ram_part=None, pinned_memory_size=None, allow_writing_files=None, final_ctr_computation_mode=None, approx_on_full_history=None, boosting_type=None, simple_ctr=None, combinations_ctr=None, per_feature_ctr=None, ctr_description=None, ctr_target_border_count=None, task_type=None, device_config=None, devices=None, bootstrap_type=None, subsample=None, mvs_reg=None, sampling_unit=None, sampling_frequency=None, dev_score_calc_obj_block_size=None, dev_efb_max_buckets=None, sparse_features_conflict_fraction=None, max_depth=None, n_estimators=None, num_boost_round=None, num_trees=None, colsample_bylevel=None, random_state=None, reg_lambda=None, objective=None, eta=None, max_bin=None, scale_pos_weight=None, gpu_cat_features_storage=None, data_partition=None, metadata=None, early_stopping_rounds=None, cat_features=None, grow_policy=None, min_data_in_leaf=None, min_child_samples=None, max_leaves=None, num_leaves=None, score_function=None, leaf_estimation_backtracking=None, ctr_history_unit=None, monotone_constraints=None, feature_weights=None, penalties_coefficient=None, first_feature_use_penalties=None, per_object_feature_penalties=None, model_shrink_rate=None, model_shrink_mode=None, langevin=None, diffusion_temperature=None, posterior_sampling=None, boost_from_average=None, text_features=None, tokenizers=None, dictionaries=None, feature_calcers=None, text_processing=None, embedding_features=None, callback=None, eval_fraction=None, fixed_binary_splits=None):
        params = {}
        not_params = ['not_params', 'self', 'params', '__class__']
        for key, value in iteritems(locals().copy()):
            if key not in not_params and value is not None:
                params[key] = value
        super(CatBoostClassifier, self).__init__(params)

    def fit(self, X, y=None, cat_features=None, text_features=None, embedding_features=None, sample_weight=None, baseline=None, use_best_model=None, eval_set=None, verbose=None, logging_level=None, plot=False, plot_file=None, column_description=None, verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None, log_cout=None, log_cerr=None):
        """
        Fit the CatBoostClassifier model.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
            If not catboost.Pool, 2 dimensional Feature matrix or string - file with dataset.

        y : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels of the training data.
            If not None, can be a single- or two- dimensional array with either:
              - numerical values - for binary classification problems
              - class labels (boolean, integer or string)
            Use only if X is not catboost.Pool and does not point to a file.

        cat_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Categ columns indices.
            Use only if X is not catboost.Pool.

        text_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Text columns indices.
            Use only if X is not catboost.Pool.

        embedding_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Embedding columns indices.
            Use only if X is not catboost.Pool.

        sample_weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        baseline : list or numpy.ndarray, optional (default=None)
            If not None, giving 2 dimensional array like data.
            Use only if X is not catboost.Pool.

        use_best_model : bool, optional (default=None)
            Flag to use best model

        eval_set : catboost.Pool or list of catboost.Pool or tuple (X, y) or list [(X, y)], optional (default=None)
            Validation dataset or datasets for metrics calculation and possibly early stopping.

        metric_period : int
            Frequency of evaluating metrics.

        verbose : bool or int
            If verbose is bool, then if set to True, logging_level is set to Verbose,
            if set to False, logging_level is set to Silent.
            If verbose is int, it determines the frequency of writing metrics to output and
            logging_level is set to Verbose.

        silent : bool
            If silent is True, logging_level is set to Silent.
            If silent is False, logging_level is set to Verbose.

        logging_level : string, optional (default=None)
            Possible values:
                - 'Silent'
                - 'Verbose'
                - 'Info'
                - 'Debug'

        plot : bool, optional (default=False)
            If True, draw train and eval error in Jupyter notebook

        plot_file : file-like or str, optional (default=None)
            If not None, save train and eval error graphs to file

        verbose_eval : bool or int
            Synonym for verbose. Only one of these parameters should be set.

        early_stopping_rounds : int
            Activates Iter overfitting detector with od_wait set to early_stopping_rounds.

        save_snapshot : bool, [default=None]
            Enable progress snapshotting for restoring progress after crashes or interruptions

        snapshot_file : string or pathlib.Path, [default=None]
            Learn progress snapshot file path, if None will use default filename

        snapshot_interval: int, [default=600]
            Interval between saving snapshots (seconds)

        init_model : CatBoost class or string or pathlib.Path, [default=None]
            Continue training starting from the existing model.
            If this parameter is a string or pathlib.Path, load initial model from the path specified by this string.

        callbacks : list, optional (default=None)
            List of callback objects that are applied at end of each iteration.

        log_cout: output stream or callback for logging (default=None)
            If None is specified, sys.stdout is used

        log_cerr: error stream or callback for logging (default=None)
            If None is specified, sys.stderr is used

        Returns
        -------
        model : CatBoost
        """
        params = self._init_params.copy()
        _process_synonyms(params)
        if 'loss_function' in params:
            CatBoostClassifier._check_is_compatible_loss(params['loss_function'])
        self._fit(X, y, cat_features, text_features, embedding_features, None, sample_weight, None, None, None, None, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
        return self

    def predict(self, data, prediction_type='Class', ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type='CPU'):
        """
        Predict with data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='Class')
            Can be:
            - 'RawFormulaVal' : return raw formula value.
            - 'Class' : return class label.
            - 'Probability' : return probability for every class.
            - 'LogProbability' : return log probability for every class.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool, optional (default=False)
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction:
            If data is for a single object, the return value depends on prediction_type value:
                - 'RawFormulaVal' : return raw formula value.
                - 'Class' : return class label.
                - 'Probability' : return one-dimensional numpy.ndarray with probability for every class.
                - 'LogProbability' : return one-dimensional numpy.ndarray with
                  log probability for every class.
            otherwise numpy.ndarray, with values that depend on prediction_type value:
                - 'RawFormulaVal' : one-dimensional array of raw formula value for each object.
                - 'Class' : one-dimensional array of class label for each object.
                - 'Probability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with probability for every class for each object.
                - 'LogProbability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with log probability for every class for each object.
        """
        return self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, 'predict', task_type)

    def predict_proba(self, X, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type='CPU'):
        """
        Predict class probability with X.

        Parameters
        ----------
        X : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If X is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction :
            If X is for a single object
                return one-dimensional numpy.ndarray with probability for every class.
            otherwise
                return two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                with probability for every class for each object.
        """
        return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)

    def predict_log_proba(self, data, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type='CPU'):
        """
        Predict class log probability with data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction :
            If data is for a single object
                return one-dimensional numpy.ndarray with log probability for every class.
            otherwise
                return two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                with log probability for every class for each object.
        """
        return self._predict(data, 'LogProbability', ntree_start, ntree_end, thread_count, verbose, 'predict_log_proba', task_type)

    def staged_predict(self, data, prediction_type='Class', ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict target at each stage for data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='Class')
            Can be:
            - 'RawFormulaVal' : return raw formula value.
            - 'Class' : return class label.
            - 'Probability' : return probability for every class.
            - 'LogProbability' : return log probability for every class.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator for each iteration that generates:
            If data is for a single object, the return value depends on prediction_type value:
                - 'RawFormulaVal' : return raw formula value.
                - 'Class' : return majority vote class.
                - 'Probability' : return one-dimensional numpy.ndarray with probability for every class.
                - 'LogProbability' : return one-dimensional numpy.ndarray with
                  log probability for every class.
            otherwise numpy.ndarray, with values that depend on prediction_type value:
                - 'RawFormulaVal' : one-dimensional array of raw formula value for each object.
                - 'Class' : one-dimensional array of class label for each object.
                - 'Probability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with probability for every class for each object.
                - 'LogProbability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with log probability for every class for each object.
        """
        return self._staged_predict(data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict')

    def staged_predict_proba(self, data, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict classification target at each stage for data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator for each iteration that generates:
            If data is for a single object
                return one-dimensional numpy.ndarray with probability for every class.
            otherwise
                return two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                with probability for every class for each object.
        """
        return self._staged_predict(data, 'Probability', ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict_proba')

    def staged_predict_log_proba(self, data, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict classification target at each stage for data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator for each iteration that generates:
            If data is for a single object
                return one-dimensional numpy.ndarray with log probability for every class.
            otherwise
                return two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                with log probability for every class for each object.
        """
        return self._staged_predict(data, 'LogProbability', ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict_log_proba')

    def score(self, X, y=None):
        """
        Calculate accuracy.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
            Data to apply model on.
        y : list or numpy.ndarray
            True labels.

        Returns
        -------
        accuracy : float
        """
        if isinstance(X, Pool):
            if y is not None:
                raise CatBoostError('Wrong initializing y: X is catboost.Pool object, y must be initialized inside catboost.Pool.')
            y = X.get_label()
            if y is None:
                raise CatBoostError('Label in X has not initialized.')
        if isinstance(y, DataFrame):
            if len(y.columns) != 1:
                raise CatBoostError('y is DataFrame and has {} columns, but must have exactly one.'.format(len(y.columns)))
            y = y[y.columns[0]]
        elif y is None:
            raise CatBoostError('y should be specified.')
        y = np.array(y)
        predicted_classes = self._predict(X, prediction_type='Class', ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, parent_method_name='score').reshape(-1)
        if np.issubdtype(predicted_classes.dtype, np.number):
            if np.issubdtype(y.dtype, np.character):
                raise CatBoostError('predicted classes have numeric type but specified y contains strings')
        elif predicted_classes.dtype == np.bool_:
            if np.issubdtype(y.dtype, np.character):
                raise CatBoostError('predicted classes have boolean type but specified y contains strings')
        elif np.issubdtype(y.dtype, np.number):
            raise CatBoostError('predicted classes have string type but specified y is numeric')
        elif np.issubdtype(y.dtype, np.bool_):
            raise CatBoostError('predicted classes have string type but specified y is boolean')
        return np.mean(np.array(predicted_classes) == np.array(y))

    def set_probability_threshold(self, binclass_probability_threshold=None):
        """
        Set a threshold for class separation in binary classification task for a trained model.
        :param binclass_probability_threshold: float number in [0, 1] or None to discard it
        """
        if not self.is_fitted():
            raise CatBoostError("You can't set probability threshold for not fitted model.")
        metadata = self.get_metadata()
        if binclass_probability_threshold is None:
            if 'binclass_probability_threshold' in metadata.keys():
                del metadata['binclass_probability_threshold']
        else:
            if not isinstance(binclass_probability_threshold, FLOAT_TYPES):
                raise CatBoostError('binclass_probability_threshold must have float type')
            assert 0.0 <= binclass_probability_threshold <= 1.0, 'Please provide correct probability for binclass_probability_threshold argument in [0, 1] range'
            self.get_metadata()['binclass_probability_threshold'] = str(binclass_probability_threshold)

    def get_probability_threshold(self):
        """
        Get a threshold for class separation in binary classification task
        """
        if not self.is_fitted():
            raise CatBoostError("Not fitted models don't have a probability threshold.")
        return self._object._get_binclass_probability_threshold()

    @staticmethod
    def _check_is_compatible_loss(loss_function):
        if isinstance(loss_function, str) and (not CatBoost._is_classification_objective(loss_function)):
            raise CatBoostError("Invalid loss_function='{}': for classifier use Logloss, CrossEntropy, MultiClass, MultiClassOneVsAll or custom objective object".format(loss_function))