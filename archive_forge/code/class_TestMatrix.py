from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestMatrix(_messages.Message):
    """TestMatrix captures all details about a test. It contains the
  environment configuration, test specification, test executions and overall
  state and outcome.

  Enums:
    InvalidMatrixDetailsValueValuesEnum: Output only. Describes why the matrix
      is considered invalid. Only useful for matrices in the INVALID state.
    OutcomeSummaryValueValuesEnum: Output Only. The overall outcome of the
      test. Only set when the test matrix state is FINISHED.
    StateValueValuesEnum: Output only. Indicates the current progress of the
      test matrix.

  Fields:
    clientInfo: Information about the client which invoked the test.
    environmentMatrix: Required. The devices the tests are being executed on.
    extendedInvalidMatrixDetails: Output only. Details about why a matrix was
      deemed invalid. If multiple checks can be safely performed, they will be
      reported but no assumptions should be made about the length of this
      list.
    failFast: If true, only a single attempt at most will be made to run each
      execution/shard in the matrix. Flaky test attempts are not affected.
      Normally, 2 or more attempts are made if a potential infrastructure
      issue is detected. This feature is for latency sensitive workloads. The
      incidence of execution failures may be significantly greater for fail-
      fast matrices and support is more limited because of that expectation.
    flakyTestAttempts: The number of times a TestExecution should be re-
      attempted if one or more of its test cases fail for any reason. The
      maximum number of reruns allowed is 10. Default is 0, which implies no
      reruns.
    invalidMatrixDetails: Output only. Describes why the matrix is considered
      invalid. Only useful for matrices in the INVALID state.
    outcomeSummary: Output Only. The overall outcome of the test. Only set
      when the test matrix state is FINISHED.
    projectId: The cloud project that owns the test matrix.
    resultStorage: Required. Where the results for the matrix are written.
    state: Output only. Indicates the current progress of the test matrix.
    testExecutions: Output only. The list of test executions that the service
      creates for this matrix.
    testMatrixId: Output only. Unique id set by the service.
    testSpecification: Required. How to run the test.
    timestamp: Output only. The time this test matrix was initially created.
  """

    class InvalidMatrixDetailsValueValuesEnum(_messages.Enum):
        """Output only. Describes why the matrix is considered invalid. Only
    useful for matrices in the INVALID state.

    Values:
      INVALID_MATRIX_DETAILS_UNSPECIFIED: Do not use. For proto versioning
        only.
      DETAILS_UNAVAILABLE: The matrix is INVALID, but there are no further
        details available.
      MALFORMED_APK: The input app APK could not be parsed.
      MALFORMED_TEST_APK: The input test APK could not be parsed.
      NO_MANIFEST: The AndroidManifest.xml could not be found.
      NO_PACKAGE_NAME: The APK manifest does not declare a package name.
      INVALID_PACKAGE_NAME: The APK application ID (aka package name) is
        invalid. See also
        https://developer.android.com/studio/build/application-id
      TEST_SAME_AS_APP: The test package and app package are the same.
      NO_INSTRUMENTATION: The test apk does not declare an instrumentation.
      NO_SIGNATURE: The input app apk does not have a signature.
      INSTRUMENTATION_ORCHESTRATOR_INCOMPATIBLE: The test runner class
        specified by user or in the test APK's manifest file is not compatible
        with Android Test Orchestrator. Orchestrator is only compatible with
        AndroidJUnitRunner version 1.1 or higher. Orchestrator can be disabled
        by using DO_NOT_USE_ORCHESTRATOR OrchestratorOption.
      NO_TEST_RUNNER_CLASS: The test APK does not contain the test runner
        class specified by the user or in the manifest file. This can be
        caused by one of the following reasons: - the user provided a runner
        class name that's incorrect, or - the test runner isn't built into the
        test APK (might be in the app APK instead).
      NO_LAUNCHER_ACTIVITY: A main launcher activity could not be found.
      FORBIDDEN_PERMISSIONS: The app declares one or more permissions that are
        not allowed.
      INVALID_ROBO_DIRECTIVES: There is a conflict in the provided
        robo_directives.
      INVALID_RESOURCE_NAME: There is at least one invalid resource name in
        the provided robo directives
      INVALID_DIRECTIVE_ACTION: Invalid definition of action in the robo
        directives (e.g. a click or ignore action includes an input text
        field)
      TEST_LOOP_INTENT_FILTER_NOT_FOUND: There is no test loop intent filter,
        or the one that is given is not formatted correctly.
      SCENARIO_LABEL_NOT_DECLARED: The request contains a scenario label that
        was not declared in the manifest.
      SCENARIO_LABEL_MALFORMED: There was an error when parsing a label's
        value.
      SCENARIO_NOT_DECLARED: The request contains a scenario number that was
        not declared in the manifest.
      DEVICE_ADMIN_RECEIVER: Device administrator applications are not
        allowed.
      MALFORMED_XC_TEST_ZIP: The zipped XCTest was malformed. The zip did not
        contain a single .xctestrun file and the contents of the
        DerivedData/Build/Products directory.
      BUILT_FOR_IOS_SIMULATOR: The zipped XCTest was built for the iOS
        simulator rather than for a physical device.
      NO_TESTS_IN_XC_TEST_ZIP: The .xctestrun file did not specify any test
        targets.
      USE_DESTINATION_ARTIFACTS: One or more of the test targets defined in
        the .xctestrun file specifies "UseDestinationArtifacts", which is
        disallowed.
      TEST_NOT_APP_HOSTED: XC tests which run on physical devices must have
        "IsAppHostedTestBundle" == "true" in the xctestrun file.
      PLIST_CANNOT_BE_PARSED: An Info.plist file in the XCTest zip could not
        be parsed.
      TEST_ONLY_APK: The APK is marked as "testOnly". Deprecated and not
        currently used.
      MALFORMED_IPA: The input IPA could not be parsed.
      MISSING_URL_SCHEME: The application doesn't register the game loop URL
        scheme.
      MALFORMED_APP_BUNDLE: The iOS application bundle (.app) couldn't be
        processed.
      NO_CODE_APK: APK contains no code. See also
        https://developer.android.com/guide/topics/manifest/application-
        element.html#code
      INVALID_INPUT_APK: Either the provided input APK path was malformed, the
        APK file does not exist, or the user does not have permission to
        access the APK file.
      INVALID_APK_PREVIEW_SDK: APK is built for a preview SDK which is
        unsupported
      MATRIX_TOO_LARGE: The matrix expanded to contain too many executions.
      TEST_QUOTA_EXCEEDED: Not enough test quota to run the executions in this
        matrix.
      SERVICE_NOT_ACTIVATED: A required cloud service api is not activated.
        See: https://firebase.google.com/docs/test-
        lab/android/continuous#requirements
      UNKNOWN_PERMISSION_ERROR: There was an unknown permission issue running
        this test.
    """
        INVALID_MATRIX_DETAILS_UNSPECIFIED = 0
        DETAILS_UNAVAILABLE = 1
        MALFORMED_APK = 2
        MALFORMED_TEST_APK = 3
        NO_MANIFEST = 4
        NO_PACKAGE_NAME = 5
        INVALID_PACKAGE_NAME = 6
        TEST_SAME_AS_APP = 7
        NO_INSTRUMENTATION = 8
        NO_SIGNATURE = 9
        INSTRUMENTATION_ORCHESTRATOR_INCOMPATIBLE = 10
        NO_TEST_RUNNER_CLASS = 11
        NO_LAUNCHER_ACTIVITY = 12
        FORBIDDEN_PERMISSIONS = 13
        INVALID_ROBO_DIRECTIVES = 14
        INVALID_RESOURCE_NAME = 15
        INVALID_DIRECTIVE_ACTION = 16
        TEST_LOOP_INTENT_FILTER_NOT_FOUND = 17
        SCENARIO_LABEL_NOT_DECLARED = 18
        SCENARIO_LABEL_MALFORMED = 19
        SCENARIO_NOT_DECLARED = 20
        DEVICE_ADMIN_RECEIVER = 21
        MALFORMED_XC_TEST_ZIP = 22
        BUILT_FOR_IOS_SIMULATOR = 23
        NO_TESTS_IN_XC_TEST_ZIP = 24
        USE_DESTINATION_ARTIFACTS = 25
        TEST_NOT_APP_HOSTED = 26
        PLIST_CANNOT_BE_PARSED = 27
        TEST_ONLY_APK = 28
        MALFORMED_IPA = 29
        MISSING_URL_SCHEME = 30
        MALFORMED_APP_BUNDLE = 31
        NO_CODE_APK = 32
        INVALID_INPUT_APK = 33
        INVALID_APK_PREVIEW_SDK = 34
        MATRIX_TOO_LARGE = 35
        TEST_QUOTA_EXCEEDED = 36
        SERVICE_NOT_ACTIVATED = 37
        UNKNOWN_PERMISSION_ERROR = 38

    class OutcomeSummaryValueValuesEnum(_messages.Enum):
        """Output Only. The overall outcome of the test. Only set when the test
    matrix state is FINISHED.

    Values:
      OUTCOME_SUMMARY_UNSPECIFIED: Do not use. For proto versioning only.
      SUCCESS: The test matrix run was successful, for instance: - All the
        test cases passed. - Robo did not detect a crash of the application
        under test.
      FAILURE: A run failed, for instance: - One or more test cases failed. -
        A test timed out. - The application under test crashed.
      INCONCLUSIVE: Something unexpected happened. The run should still be
        considered unsuccessful but this is likely a transient problem and re-
        running the test might be successful.
      SKIPPED: All tests were skipped, for instance: - All device
        configurations were incompatible.
    """
        OUTCOME_SUMMARY_UNSPECIFIED = 0
        SUCCESS = 1
        FAILURE = 2
        INCONCLUSIVE = 3
        SKIPPED = 4

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Indicates the current progress of the test matrix.

    Values:
      TEST_STATE_UNSPECIFIED: Do not use. For proto versioning only.
      VALIDATING: The execution or matrix is being validated.
      PENDING: The execution or matrix is waiting for resources to become
        available.
      RUNNING: The execution is currently being processed. Can only be set on
        an execution.
      FINISHED: The execution or matrix has terminated normally. On a matrix
        this means that the matrix level processing completed normally, but
        individual executions may be in an ERROR state.
      ERROR: The execution or matrix has stopped because it encountered an
        infrastructure failure.
      UNSUPPORTED_ENVIRONMENT: The execution was not run because it
        corresponds to a unsupported environment. Can only be set on an
        execution.
      INCOMPATIBLE_ENVIRONMENT: The execution was not run because the provided
        inputs are incompatible with the requested environment. Example:
        requested AndroidVersion is lower than APK's minSdkVersion Can only be
        set on an execution.
      INCOMPATIBLE_ARCHITECTURE: The execution was not run because the
        provided inputs are incompatible with the requested architecture.
        Example: requested device does not support running the native code in
        the supplied APK Can only be set on an execution.
      CANCELLED: The user cancelled the execution. Can only be set on an
        execution.
      INVALID: The execution or matrix was not run because the provided inputs
        are not valid. Examples: input file is not of the expected type, is
        malformed/corrupt, or was flagged as malware
    """
        TEST_STATE_UNSPECIFIED = 0
        VALIDATING = 1
        PENDING = 2
        RUNNING = 3
        FINISHED = 4
        ERROR = 5
        UNSUPPORTED_ENVIRONMENT = 6
        INCOMPATIBLE_ENVIRONMENT = 7
        INCOMPATIBLE_ARCHITECTURE = 8
        CANCELLED = 9
        INVALID = 10
    clientInfo = _messages.MessageField('ClientInfo', 1)
    environmentMatrix = _messages.MessageField('EnvironmentMatrix', 2)
    extendedInvalidMatrixDetails = _messages.MessageField('MatrixErrorDetail', 3, repeated=True)
    failFast = _messages.BooleanField(4)
    flakyTestAttempts = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    invalidMatrixDetails = _messages.EnumField('InvalidMatrixDetailsValueValuesEnum', 6)
    outcomeSummary = _messages.EnumField('OutcomeSummaryValueValuesEnum', 7)
    projectId = _messages.StringField(8)
    resultStorage = _messages.MessageField('ResultStorage', 9)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    testExecutions = _messages.MessageField('TestExecution', 11, repeated=True)
    testMatrixId = _messages.StringField(12)
    testSpecification = _messages.MessageField('TestSpecification', 13)
    timestamp = _messages.StringField(14)