from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrainingOptions(_messages.Message):
    """Options used in model training.

  Enums:
    BoosterTypeValueValuesEnum: Booster type for boosted tree models.
    CategoryEncodingMethodValueValuesEnum: Categorical feature encoding
      method.
    ColorSpaceValueValuesEnum: Enums for color space, used for processing
      images in Object Table. See more details at
      https://www.tensorflow.org/io/tutorials/colorspace.
    DartNormalizeTypeValueValuesEnum: Type of normalization algorithm for
      boosted tree models using dart booster.
    DataFrequencyValueValuesEnum: The data frequency of a time series.
    DataSplitMethodValueValuesEnum: The data split type for training and
      evaluation, e.g. RANDOM.
    DistanceTypeValueValuesEnum: Distance type for clustering models.
    FeedbackTypeValueValuesEnum: Feedback type that specifies which algorithm
      to run for matrix factorization.
    HolidayRegionValueValuesEnum: The geographical region based on which the
      holidays are considered in time series modeling. If a valid value is
      specified, then holiday effects modeling is enabled.
    HolidayRegionsValueListEntryValuesEnum:
    HparamTuningObjectivesValueListEntryValuesEnum:
    KmeansInitializationMethodValueValuesEnum: The method used to initialize
      the centroids for kmeans algorithm.
    LearnRateStrategyValueValuesEnum: The strategy to determine learn rate for
      the current iteration.
    LossTypeValueValuesEnum: Type of loss function used during training run.
    ModelRegistryValueValuesEnum: The model registry.
    OptimizationStrategyValueValuesEnum: Optimization strategy for training
      linear regression models.
    PcaSolverValueValuesEnum: The solver for PCA.
    TreeMethodValueValuesEnum: Tree construction algorithm for boosted tree
      models.

  Messages:
    LabelClassWeightsValue: Weights associated with each label class, for
      rebalancing the training data. Only applicable for classification
      models.

  Fields:
    activationFn: Activation function of the neural nets.
    adjustStepChanges: If true, detect step changes and make data adjustment
      in the input time series.
    approxGlobalFeatureContrib: Whether to use approximate feature
      contribution method in XGBoost model explanation for global explain.
    autoArima: Whether to enable auto ARIMA or not.
    autoArimaMaxOrder: The max value of the sum of non-seasonal p and q.
    autoArimaMinOrder: The min value of the sum of non-seasonal p and q.
    autoClassWeights: Whether to calculate class weights automatically based
      on the popularity of each label.
    batchSize: Batch size for dnn models.
    boosterType: Booster type for boosted tree models.
    budgetHours: Budget in hours for AutoML training.
    calculatePValues: Whether or not p-value test should be computed for this
      model. Only available for linear and logistic regression models.
    categoryEncodingMethod: Categorical feature encoding method.
    cleanSpikesAndDips: If true, clean spikes and dips in the input time
      series.
    colorSpace: Enums for color space, used for processing images in Object
      Table. See more details at
      https://www.tensorflow.org/io/tutorials/colorspace.
    colsampleBylevel: Subsample ratio of columns for each level for boosted
      tree models.
    colsampleBynode: Subsample ratio of columns for each node(split) for
      boosted tree models.
    colsampleBytree: Subsample ratio of columns when constructing each tree
      for boosted tree models.
    dartNormalizeType: Type of normalization algorithm for boosted tree models
      using dart booster.
    dataFrequency: The data frequency of a time series.
    dataSplitColumn: The column to split data with. This column won't be used
      as a feature. 1. When data_split_method is CUSTOM, the corresponding
      column should be boolean. The rows with true value tag are eval data,
      and the false are training data. 2. When data_split_method is SEQ, the
      first DATA_SPLIT_EVAL_FRACTION rows (from smallest to largest) in the
      corresponding column are used as training data, and the rest are eval
      data. It respects the order in Orderable data types:
      https://cloud.google.com/bigquery/docs/reference/standard-sql/data-
      types#data-type-properties
    dataSplitEvalFraction: The fraction of evaluation data over the whole
      input data. The rest of data will be used as training data. The format
      should be double. Accurate to two decimal places. Default value is 0.2.
    dataSplitMethod: The data split type for training and evaluation, e.g.
      RANDOM.
    decomposeTimeSeries: If true, perform decompose time series and save the
      results.
    distanceType: Distance type for clustering models.
    dropout: Dropout probability for dnn models.
    earlyStop: Whether to stop early when the loss doesn't improve
      significantly any more (compared to min_relative_progress). Used only
      for iterative training algorithms.
    enableGlobalExplain: If true, enable global explanation during training.
    feedbackType: Feedback type that specifies which algorithm to run for
      matrix factorization.
    fitIntercept: Whether the model should include intercept during model
      training.
    hiddenUnits: Hidden units for dnn models.
    holidayRegion: The geographical region based on which the holidays are
      considered in time series modeling. If a valid value is specified, then
      holiday effects modeling is enabled.
    holidayRegions: A list of geographical regions that are used for time
      series modeling.
    horizon: The number of periods ahead that need to be forecasted.
    hparamTuningObjectives: The target evaluation metrics to optimize the
      hyperparameters for.
    includeDrift: Include drift when fitting an ARIMA model.
    initialLearnRate: Specifies the initial learning rate for the line search
      learn rate strategy.
    inputLabelColumns: Name of input label columns in training data.
    instanceWeightColumn: Name of the instance weight column for training
      data. This column isn't be used as a feature.
    integratedGradientsNumSteps: Number of integral steps for the integrated
      gradients explain method.
    itemColumn: Item column specified for matrix factorization models.
    kmeansInitializationColumn: The column used to provide the initial
      centroids for kmeans algorithm when kmeans_initialization_method is
      CUSTOM.
    kmeansInitializationMethod: The method used to initialize the centroids
      for kmeans algorithm.
    l1RegActivation: L1 regularization coefficient to activations.
    l1Regularization: L1 regularization coefficient.
    l2Regularization: L2 regularization coefficient.
    labelClassWeights: Weights associated with each label class, for
      rebalancing the training data. Only applicable for classification
      models.
    learnRate: Learning rate in training. Used only for iterative training
      algorithms.
    learnRateStrategy: The strategy to determine learn rate for the current
      iteration.
    lossType: Type of loss function used during training run.
    maxIterations: The maximum number of iterations in training. Used only for
      iterative training algorithms.
    maxParallelTrials: Maximum number of trials to run in parallel.
    maxTimeSeriesLength: The maximum number of time points in a time series
      that can be used in modeling the trend component of the time series.
      Don't use this option with the `timeSeriesLengthFraction` or
      `minTimeSeriesLength` options.
    maxTreeDepth: Maximum depth of a tree for boosted tree models.
    minRelativeProgress: When early_stop is true, stops training when accuracy
      improvement is less than 'min_relative_progress'. Used only for
      iterative training algorithms.
    minSplitLoss: Minimum split loss for boosted tree models.
    minTimeSeriesLength: The minimum number of time points in a time series
      that are used in modeling the trend component of the time series. If you
      use this option you must also set the `timeSeriesLengthFraction` option.
      This training option ensures that enough time points are available when
      you use `timeSeriesLengthFraction` in trend modeling. This is
      particularly important when forecasting multiple time series in a single
      query using `timeSeriesIdColumn`. If the total number of time points is
      less than the `minTimeSeriesLength` value, then the query uses all
      available time points.
    minTreeChildWeight: Minimum sum of instance weight needed in a child for
      boosted tree models.
    modelRegistry: The model registry.
    modelUri: Google Cloud Storage URI from which the model was imported. Only
      applicable for imported models.
    nonSeasonalOrder: A specification of the non-seasonal part of the ARIMA
      model: the three components (p, d, q) are the AR order, the degree of
      differencing, and the MA order.
    numClusters: Number of clusters for clustering models.
    numFactors: Num factors specified for matrix factorization models.
    numParallelTree: Number of parallel trees constructed during each
      iteration for boosted tree models.
    numPrincipalComponents: Number of principal components to keep in the PCA
      model. Must be <= the number of features.
    numTrials: Number of trials to run this hyperparameter tuning job.
    optimizationStrategy: Optimization strategy for training linear regression
      models.
    optimizer: Optimizer used for training the neural nets.
    pcaExplainedVarianceRatio: The minimum ratio of cumulative explained
      variance that needs to be given by the PCA model.
    pcaSolver: The solver for PCA.
    sampledShapleyNumPaths: Number of paths for the sampled Shapley explain
      method.
    scaleFeatures: If true, scale the feature values by dividing the feature
      standard deviation. Currently only apply to PCA.
    standardizeFeatures: Whether to standardize numerical features. Default to
      true.
    subsample: Subsample fraction of the training data to grow tree to prevent
      overfitting for boosted tree models.
    tfVersion: Based on the selected TF version, the corresponding docker
      image is used to train external models.
    timeSeriesDataColumn: Column to be designated as time series data for
      ARIMA model.
    timeSeriesIdColumn: The time series id column that was used during ARIMA
      model training.
    timeSeriesIdColumns: The time series id columns that were used during
      ARIMA model training.
    timeSeriesLengthFraction: The fraction of the interpolated length of the
      time series that's used to model the time series trend component. All of
      the time points of the time series are used to model the non-trend
      component. This training option accelerates modeling training without
      sacrificing much forecasting accuracy. You can use this option with
      `minTimeSeriesLength` but not with `maxTimeSeriesLength`.
    timeSeriesTimestampColumn: Column to be designated as time series
      timestamp for ARIMA model.
    treeMethod: Tree construction algorithm for boosted tree models.
    trendSmoothingWindowSize: Smoothing window size for the trend component.
      When a positive value is specified, a center moving average smoothing is
      applied on the history trend. When the smoothing window is out of the
      boundary at the beginning or the end of the trend, the first element or
      the last element is padded to fill the smoothing window before the
      average is applied.
    userColumn: User column specified for matrix factorization models.
    vertexAiModelVersionAliases: The version aliases to apply in Vertex AI
      model registry. Always overwrite if the version aliases exists in a
      existing model.
    walsAlpha: Hyperparameter for matrix factoration when implicit feedback
      type is specified.
    warmStart: Whether to train a model from the last checkpoint.
    xgboostVersion: User-selected XGBoost versions for training of XGBoost
      models.
  """

    class BoosterTypeValueValuesEnum(_messages.Enum):
        """Booster type for boosted tree models.

    Values:
      BOOSTER_TYPE_UNSPECIFIED: Unspecified booster type.
      GBTREE: Gbtree booster.
      DART: Dart booster.
    """
        BOOSTER_TYPE_UNSPECIFIED = 0
        GBTREE = 1
        DART = 2

    class CategoryEncodingMethodValueValuesEnum(_messages.Enum):
        """Categorical feature encoding method.

    Values:
      ENCODING_METHOD_UNSPECIFIED: Unspecified encoding method.
      ONE_HOT_ENCODING: Applies one-hot encoding.
      LABEL_ENCODING: Applies label encoding.
      DUMMY_ENCODING: Applies dummy encoding.
    """
        ENCODING_METHOD_UNSPECIFIED = 0
        ONE_HOT_ENCODING = 1
        LABEL_ENCODING = 2
        DUMMY_ENCODING = 3

    class ColorSpaceValueValuesEnum(_messages.Enum):
        """Enums for color space, used for processing images in Object Table. See
    more details at https://www.tensorflow.org/io/tutorials/colorspace.

    Values:
      COLOR_SPACE_UNSPECIFIED: Unspecified color space
      RGB: RGB
      HSV: HSV
      YIQ: YIQ
      YUV: YUV
      GRAYSCALE: GRAYSCALE
    """
        COLOR_SPACE_UNSPECIFIED = 0
        RGB = 1
        HSV = 2
        YIQ = 3
        YUV = 4
        GRAYSCALE = 5

    class DartNormalizeTypeValueValuesEnum(_messages.Enum):
        """Type of normalization algorithm for boosted tree models using dart
    booster.

    Values:
      DART_NORMALIZE_TYPE_UNSPECIFIED: Unspecified dart normalize type.
      TREE: New trees have the same weight of each of dropped trees.
      FOREST: New trees have the same weight of sum of dropped trees.
    """
        DART_NORMALIZE_TYPE_UNSPECIFIED = 0
        TREE = 1
        FOREST = 2

    class DataFrequencyValueValuesEnum(_messages.Enum):
        """The data frequency of a time series.

    Values:
      DATA_FREQUENCY_UNSPECIFIED: Default value.
      AUTO_FREQUENCY: Automatically inferred from timestamps.
      YEARLY: Yearly data.
      QUARTERLY: Quarterly data.
      MONTHLY: Monthly data.
      WEEKLY: Weekly data.
      DAILY: Daily data.
      HOURLY: Hourly data.
      PER_MINUTE: Per-minute data.
    """
        DATA_FREQUENCY_UNSPECIFIED = 0
        AUTO_FREQUENCY = 1
        YEARLY = 2
        QUARTERLY = 3
        MONTHLY = 4
        WEEKLY = 5
        DAILY = 6
        HOURLY = 7
        PER_MINUTE = 8

    class DataSplitMethodValueValuesEnum(_messages.Enum):
        """The data split type for training and evaluation, e.g. RANDOM.

    Values:
      DATA_SPLIT_METHOD_UNSPECIFIED: Default value.
      RANDOM: Splits data randomly.
      CUSTOM: Splits data with the user provided tags.
      SEQUENTIAL: Splits data sequentially.
      NO_SPLIT: Data split will be skipped.
      AUTO_SPLIT: Splits data automatically: Uses NO_SPLIT if the data size is
        small. Otherwise uses RANDOM.
    """
        DATA_SPLIT_METHOD_UNSPECIFIED = 0
        RANDOM = 1
        CUSTOM = 2
        SEQUENTIAL = 3
        NO_SPLIT = 4
        AUTO_SPLIT = 5

    class DistanceTypeValueValuesEnum(_messages.Enum):
        """Distance type for clustering models.

    Values:
      DISTANCE_TYPE_UNSPECIFIED: Default value.
      EUCLIDEAN: Eculidean distance.
      COSINE: Cosine distance.
    """
        DISTANCE_TYPE_UNSPECIFIED = 0
        EUCLIDEAN = 1
        COSINE = 2

    class FeedbackTypeValueValuesEnum(_messages.Enum):
        """Feedback type that specifies which algorithm to run for matrix
    factorization.

    Values:
      FEEDBACK_TYPE_UNSPECIFIED: Default value.
      IMPLICIT: Use weighted-als for implicit feedback problems.
      EXPLICIT: Use nonweighted-als for explicit feedback problems.
    """
        FEEDBACK_TYPE_UNSPECIFIED = 0
        IMPLICIT = 1
        EXPLICIT = 2

    class HolidayRegionValueValuesEnum(_messages.Enum):
        """The geographical region based on which the holidays are considered in
    time series modeling. If a valid value is specified, then holiday effects
    modeling is enabled.

    Values:
      HOLIDAY_REGION_UNSPECIFIED: Holiday region unspecified.
      GLOBAL: Global.
      NA: North America.
      JAPAC: Japan and Asia Pacific: Korea, Greater China, India, Australia,
        and New Zealand.
      EMEA: Europe, the Middle East and Africa.
      LAC: Latin America and the Caribbean.
      AE: United Arab Emirates
      AR: Argentina
      AT: Austria
      AU: Australia
      BE: Belgium
      BR: Brazil
      CA: Canada
      CH: Switzerland
      CL: Chile
      CN: China
      CO: Colombia
      CS: Czechoslovakia
      CZ: Czech Republic
      DE: Germany
      DK: Denmark
      DZ: Algeria
      EC: Ecuador
      EE: Estonia
      EG: Egypt
      ES: Spain
      FI: Finland
      FR: France
      GB: Great Britain (United Kingdom)
      GR: Greece
      HK: Hong Kong
      HU: Hungary
      ID: Indonesia
      IE: Ireland
      IL: Israel
      IN: India
      IR: Iran
      IT: Italy
      JP: Japan
      KR: Korea (South)
      LV: Latvia
      MA: Morocco
      MX: Mexico
      MY: Malaysia
      NG: Nigeria
      NL: Netherlands
      NO: Norway
      NZ: New Zealand
      PE: Peru
      PH: Philippines
      PK: Pakistan
      PL: Poland
      PT: Portugal
      RO: Romania
      RS: Serbia
      RU: Russian Federation
      SA: Saudi Arabia
      SE: Sweden
      SG: Singapore
      SI: Slovenia
      SK: Slovakia
      TH: Thailand
      TR: Turkey
      TW: Taiwan
      UA: Ukraine
      US: United States
      VE: Venezuela
      VN: Viet Nam
      ZA: South Africa
    """
        HOLIDAY_REGION_UNSPECIFIED = 0
        GLOBAL = 1
        NA = 2
        JAPAC = 3
        EMEA = 4
        LAC = 5
        AE = 6
        AR = 7
        AT = 8
        AU = 9
        BE = 10
        BR = 11
        CA = 12
        CH = 13
        CL = 14
        CN = 15
        CO = 16
        CS = 17
        CZ = 18
        DE = 19
        DK = 20
        DZ = 21
        EC = 22
        EE = 23
        EG = 24
        ES = 25
        FI = 26
        FR = 27
        GB = 28
        GR = 29
        HK = 30
        HU = 31
        ID = 32
        IE = 33
        IL = 34
        IN = 35
        IR = 36
        IT = 37
        JP = 38
        KR = 39
        LV = 40
        MA = 41
        MX = 42
        MY = 43
        NG = 44
        NL = 45
        NO = 46
        NZ = 47
        PE = 48
        PH = 49
        PK = 50
        PL = 51
        PT = 52
        RO = 53
        RS = 54
        RU = 55
        SA = 56
        SE = 57
        SG = 58
        SI = 59
        SK = 60
        TH = 61
        TR = 62
        TW = 63
        UA = 64
        US = 65
        VE = 66
        VN = 67
        ZA = 68

    class HolidayRegionsValueListEntryValuesEnum(_messages.Enum):
        """HolidayRegionsValueListEntryValuesEnum enum type.

    Values:
      HOLIDAY_REGION_UNSPECIFIED: Holiday region unspecified.
      GLOBAL: Global.
      NA: North America.
      JAPAC: Japan and Asia Pacific: Korea, Greater China, India, Australia,
        and New Zealand.
      EMEA: Europe, the Middle East and Africa.
      LAC: Latin America and the Caribbean.
      AE: United Arab Emirates
      AR: Argentina
      AT: Austria
      AU: Australia
      BE: Belgium
      BR: Brazil
      CA: Canada
      CH: Switzerland
      CL: Chile
      CN: China
      CO: Colombia
      CS: Czechoslovakia
      CZ: Czech Republic
      DE: Germany
      DK: Denmark
      DZ: Algeria
      EC: Ecuador
      EE: Estonia
      EG: Egypt
      ES: Spain
      FI: Finland
      FR: France
      GB: Great Britain (United Kingdom)
      GR: Greece
      HK: Hong Kong
      HU: Hungary
      ID: Indonesia
      IE: Ireland
      IL: Israel
      IN: India
      IR: Iran
      IT: Italy
      JP: Japan
      KR: Korea (South)
      LV: Latvia
      MA: Morocco
      MX: Mexico
      MY: Malaysia
      NG: Nigeria
      NL: Netherlands
      NO: Norway
      NZ: New Zealand
      PE: Peru
      PH: Philippines
      PK: Pakistan
      PL: Poland
      PT: Portugal
      RO: Romania
      RS: Serbia
      RU: Russian Federation
      SA: Saudi Arabia
      SE: Sweden
      SG: Singapore
      SI: Slovenia
      SK: Slovakia
      TH: Thailand
      TR: Turkey
      TW: Taiwan
      UA: Ukraine
      US: United States
      VE: Venezuela
      VN: Viet Nam
      ZA: South Africa
    """
        HOLIDAY_REGION_UNSPECIFIED = 0
        GLOBAL = 1
        NA = 2
        JAPAC = 3
        EMEA = 4
        LAC = 5
        AE = 6
        AR = 7
        AT = 8
        AU = 9
        BE = 10
        BR = 11
        CA = 12
        CH = 13
        CL = 14
        CN = 15
        CO = 16
        CS = 17
        CZ = 18
        DE = 19
        DK = 20
        DZ = 21
        EC = 22
        EE = 23
        EG = 24
        ES = 25
        FI = 26
        FR = 27
        GB = 28
        GR = 29
        HK = 30
        HU = 31
        ID = 32
        IE = 33
        IL = 34
        IN = 35
        IR = 36
        IT = 37
        JP = 38
        KR = 39
        LV = 40
        MA = 41
        MX = 42
        MY = 43
        NG = 44
        NL = 45
        NO = 46
        NZ = 47
        PE = 48
        PH = 49
        PK = 50
        PL = 51
        PT = 52
        RO = 53
        RS = 54
        RU = 55
        SA = 56
        SE = 57
        SG = 58
        SI = 59
        SK = 60
        TH = 61
        TR = 62
        TW = 63
        UA = 64
        US = 65
        VE = 66
        VN = 67
        ZA = 68

    class HparamTuningObjectivesValueListEntryValuesEnum(_messages.Enum):
        """HparamTuningObjectivesValueListEntryValuesEnum enum type.

    Values:
      HPARAM_TUNING_OBJECTIVE_UNSPECIFIED: Unspecified evaluation metric.
      MEAN_ABSOLUTE_ERROR: Mean absolute error. mean_absolute_error =
        AVG(ABS(label - predicted))
      MEAN_SQUARED_ERROR: Mean squared error. mean_squared_error =
        AVG(POW(label - predicted, 2))
      MEAN_SQUARED_LOG_ERROR: Mean squared log error. mean_squared_log_error =
        AVG(POW(LN(1 + label) - LN(1 + predicted), 2))
      MEDIAN_ABSOLUTE_ERROR: Mean absolute error. median_absolute_error =
        APPROX_QUANTILES(absolute_error, 2)[OFFSET(1)]
      R_SQUARED: R^2 score. This corresponds to r2_score in ML.EVALUATE.
        r_squared = 1 - SUM(squared_error)/(COUNT(label)*VAR_POP(label))
      EXPLAINED_VARIANCE: Explained variance. explained_variance = 1 -
        VAR_POP(label_error)/VAR_POP(label)
      PRECISION: Precision is the fraction of actual positive predictions that
        had positive actual labels. For multiclass this is a macro-averaged
        metric treating each class as a binary classifier.
      RECALL: Recall is the fraction of actual positive labels that were given
        a positive prediction. For multiclass this is a macro-averaged metric.
      ACCURACY: Accuracy is the fraction of predictions given the correct
        label. For multiclass this is a globally micro-averaged metric.
      F1_SCORE: The F1 score is an average of recall and precision. For
        multiclass this is a macro-averaged metric.
      LOG_LOSS: Logorithmic Loss. For multiclass this is a macro-averaged
        metric.
      ROC_AUC: Area Under an ROC Curve. For multiclass this is a macro-
        averaged metric.
      DAVIES_BOULDIN_INDEX: Davies-Bouldin Index.
      MEAN_AVERAGE_PRECISION: Mean Average Precision.
      NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN: Normalized Discounted Cumulative
        Gain.
      AVERAGE_RANK: Average Rank.
    """
        HPARAM_TUNING_OBJECTIVE_UNSPECIFIED = 0
        MEAN_ABSOLUTE_ERROR = 1
        MEAN_SQUARED_ERROR = 2
        MEAN_SQUARED_LOG_ERROR = 3
        MEDIAN_ABSOLUTE_ERROR = 4
        R_SQUARED = 5
        EXPLAINED_VARIANCE = 6
        PRECISION = 7
        RECALL = 8
        ACCURACY = 9
        F1_SCORE = 10
        LOG_LOSS = 11
        ROC_AUC = 12
        DAVIES_BOULDIN_INDEX = 13
        MEAN_AVERAGE_PRECISION = 14
        NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN = 15
        AVERAGE_RANK = 16

    class KmeansInitializationMethodValueValuesEnum(_messages.Enum):
        """The method used to initialize the centroids for kmeans algorithm.

    Values:
      KMEANS_INITIALIZATION_METHOD_UNSPECIFIED: Unspecified initialization
        method.
      RANDOM: Initializes the centroids randomly.
      CUSTOM: Initializes the centroids using data specified in
        kmeans_initialization_column.
      KMEANS_PLUS_PLUS: Initializes with kmeans++.
    """
        KMEANS_INITIALIZATION_METHOD_UNSPECIFIED = 0
        RANDOM = 1
        CUSTOM = 2
        KMEANS_PLUS_PLUS = 3

    class LearnRateStrategyValueValuesEnum(_messages.Enum):
        """The strategy to determine learn rate for the current iteration.

    Values:
      LEARN_RATE_STRATEGY_UNSPECIFIED: Default value.
      LINE_SEARCH: Use line search to determine learning rate.
      CONSTANT: Use a constant learning rate.
    """
        LEARN_RATE_STRATEGY_UNSPECIFIED = 0
        LINE_SEARCH = 1
        CONSTANT = 2

    class LossTypeValueValuesEnum(_messages.Enum):
        """Type of loss function used during training run.

    Values:
      LOSS_TYPE_UNSPECIFIED: Default value.
      MEAN_SQUARED_LOSS: Mean squared loss, used for linear regression.
      MEAN_LOG_LOSS: Mean log loss, used for logistic regression.
    """
        LOSS_TYPE_UNSPECIFIED = 0
        MEAN_SQUARED_LOSS = 1
        MEAN_LOG_LOSS = 2

    class ModelRegistryValueValuesEnum(_messages.Enum):
        """The model registry.

    Values:
      MODEL_REGISTRY_UNSPECIFIED: Default value.
      VERTEX_AI: Vertex AI.
    """
        MODEL_REGISTRY_UNSPECIFIED = 0
        VERTEX_AI = 1

    class OptimizationStrategyValueValuesEnum(_messages.Enum):
        """Optimization strategy for training linear regression models.

    Values:
      OPTIMIZATION_STRATEGY_UNSPECIFIED: Default value.
      BATCH_GRADIENT_DESCENT: Uses an iterative batch gradient descent
        algorithm.
      NORMAL_EQUATION: Uses a normal equation to solve linear regression
        problem.
    """
        OPTIMIZATION_STRATEGY_UNSPECIFIED = 0
        BATCH_GRADIENT_DESCENT = 1
        NORMAL_EQUATION = 2

    class PcaSolverValueValuesEnum(_messages.Enum):
        """The solver for PCA.

    Values:
      UNSPECIFIED: Default value.
      FULL: Full eigen-decoposition.
      RANDOMIZED: Randomized SVD.
      AUTO: Auto.
    """
        UNSPECIFIED = 0
        FULL = 1
        RANDOMIZED = 2
        AUTO = 3

    class TreeMethodValueValuesEnum(_messages.Enum):
        """Tree construction algorithm for boosted tree models.

    Values:
      TREE_METHOD_UNSPECIFIED: Unspecified tree method.
      AUTO: Use heuristic to choose the fastest method.
      EXACT: Exact greedy algorithm.
      APPROX: Approximate greedy algorithm using quantile sketch and gradient
        histogram.
      HIST: Fast histogram optimized approximate greedy algorithm.
    """
        TREE_METHOD_UNSPECIFIED = 0
        AUTO = 1
        EXACT = 2
        APPROX = 3
        HIST = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelClassWeightsValue(_messages.Message):
        """Weights associated with each label class, for rebalancing the training
    data. Only applicable for classification models.

    Messages:
      AdditionalProperty: An additional property for a LabelClassWeightsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        LabelClassWeightsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelClassWeightsValue object.

      Fields:
        key: Name of the additional property.
        value: A number attribute.
      """
            key = _messages.StringField(1)
            value = _messages.FloatField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    activationFn = _messages.StringField(1)
    adjustStepChanges = _messages.BooleanField(2)
    approxGlobalFeatureContrib = _messages.BooleanField(3)
    autoArima = _messages.BooleanField(4)
    autoArimaMaxOrder = _messages.IntegerField(5)
    autoArimaMinOrder = _messages.IntegerField(6)
    autoClassWeights = _messages.BooleanField(7)
    batchSize = _messages.IntegerField(8)
    boosterType = _messages.EnumField('BoosterTypeValueValuesEnum', 9)
    budgetHours = _messages.FloatField(10)
    calculatePValues = _messages.BooleanField(11)
    categoryEncodingMethod = _messages.EnumField('CategoryEncodingMethodValueValuesEnum', 12)
    cleanSpikesAndDips = _messages.BooleanField(13)
    colorSpace = _messages.EnumField('ColorSpaceValueValuesEnum', 14)
    colsampleBylevel = _messages.FloatField(15)
    colsampleBynode = _messages.FloatField(16)
    colsampleBytree = _messages.FloatField(17)
    dartNormalizeType = _messages.EnumField('DartNormalizeTypeValueValuesEnum', 18)
    dataFrequency = _messages.EnumField('DataFrequencyValueValuesEnum', 19)
    dataSplitColumn = _messages.StringField(20)
    dataSplitEvalFraction = _messages.FloatField(21)
    dataSplitMethod = _messages.EnumField('DataSplitMethodValueValuesEnum', 22)
    decomposeTimeSeries = _messages.BooleanField(23)
    distanceType = _messages.EnumField('DistanceTypeValueValuesEnum', 24)
    dropout = _messages.FloatField(25)
    earlyStop = _messages.BooleanField(26)
    enableGlobalExplain = _messages.BooleanField(27)
    feedbackType = _messages.EnumField('FeedbackTypeValueValuesEnum', 28)
    fitIntercept = _messages.BooleanField(29)
    hiddenUnits = _messages.IntegerField(30, repeated=True)
    holidayRegion = _messages.EnumField('HolidayRegionValueValuesEnum', 31)
    holidayRegions = _messages.EnumField('HolidayRegionsValueListEntryValuesEnum', 32, repeated=True)
    horizon = _messages.IntegerField(33)
    hparamTuningObjectives = _messages.EnumField('HparamTuningObjectivesValueListEntryValuesEnum', 34, repeated=True)
    includeDrift = _messages.BooleanField(35)
    initialLearnRate = _messages.FloatField(36)
    inputLabelColumns = _messages.StringField(37, repeated=True)
    instanceWeightColumn = _messages.StringField(38)
    integratedGradientsNumSteps = _messages.IntegerField(39)
    itemColumn = _messages.StringField(40)
    kmeansInitializationColumn = _messages.StringField(41)
    kmeansInitializationMethod = _messages.EnumField('KmeansInitializationMethodValueValuesEnum', 42)
    l1RegActivation = _messages.FloatField(43)
    l1Regularization = _messages.FloatField(44)
    l2Regularization = _messages.FloatField(45)
    labelClassWeights = _messages.MessageField('LabelClassWeightsValue', 46)
    learnRate = _messages.FloatField(47)
    learnRateStrategy = _messages.EnumField('LearnRateStrategyValueValuesEnum', 48)
    lossType = _messages.EnumField('LossTypeValueValuesEnum', 49)
    maxIterations = _messages.IntegerField(50)
    maxParallelTrials = _messages.IntegerField(51)
    maxTimeSeriesLength = _messages.IntegerField(52)
    maxTreeDepth = _messages.IntegerField(53)
    minRelativeProgress = _messages.FloatField(54)
    minSplitLoss = _messages.FloatField(55)
    minTimeSeriesLength = _messages.IntegerField(56)
    minTreeChildWeight = _messages.IntegerField(57)
    modelRegistry = _messages.EnumField('ModelRegistryValueValuesEnum', 58)
    modelUri = _messages.StringField(59)
    nonSeasonalOrder = _messages.MessageField('ArimaOrder', 60)
    numClusters = _messages.IntegerField(61)
    numFactors = _messages.IntegerField(62)
    numParallelTree = _messages.IntegerField(63)
    numPrincipalComponents = _messages.IntegerField(64)
    numTrials = _messages.IntegerField(65)
    optimizationStrategy = _messages.EnumField('OptimizationStrategyValueValuesEnum', 66)
    optimizer = _messages.StringField(67)
    pcaExplainedVarianceRatio = _messages.FloatField(68)
    pcaSolver = _messages.EnumField('PcaSolverValueValuesEnum', 69)
    sampledShapleyNumPaths = _messages.IntegerField(70)
    scaleFeatures = _messages.BooleanField(71)
    standardizeFeatures = _messages.BooleanField(72)
    subsample = _messages.FloatField(73)
    tfVersion = _messages.StringField(74)
    timeSeriesDataColumn = _messages.StringField(75)
    timeSeriesIdColumn = _messages.StringField(76)
    timeSeriesIdColumns = _messages.StringField(77, repeated=True)
    timeSeriesLengthFraction = _messages.FloatField(78)
    timeSeriesTimestampColumn = _messages.StringField(79)
    treeMethod = _messages.EnumField('TreeMethodValueValuesEnum', 80)
    trendSmoothingWindowSize = _messages.IntegerField(81)
    userColumn = _messages.StringField(82)
    vertexAiModelVersionAliases = _messages.StringField(83, repeated=True)
    walsAlpha = _messages.FloatField(84)
    warmStart = _messages.BooleanField(85)
    xgboostVersion = _messages.StringField(86)