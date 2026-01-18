from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import arg_util
from googlecloudsdk.api_lib.firebase.test import ctrl_c_handler
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.api_lib.firebase.test import history_picker
from googlecloudsdk.api_lib.firebase.test import matrix_ops
from googlecloudsdk.api_lib.firebase.test import results_bucket
from googlecloudsdk.api_lib.firebase.test import results_summary
from googlecloudsdk.api_lib.firebase.test import tool_results
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.api_lib.firebase.test.android import arg_manager
from googlecloudsdk.api_lib.firebase.test.android import matrix_creator
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
import six
@base.UnicodeIsSupported
class _BaseRun(object):
    """Invoke a test in Firebase Test Lab for Android and view test results."""
    detailed_help = {'DESCRIPTION': '\n          *{command}* invokes and monitors tests in Firebase Test Lab for\n          Android.\n\n          Three main types of Android tests are currently supported:\n          - *robo*: runs a smart, automated exploration of the activities in\n            your Android app which records any installation failures or crashes\n            and builds an activity map with associated screenshots and video.\n          - *instrumentation*: runs automated unit or integration tests written\n            using a testing framework. Firebase Test Lab for Android currently\n            supports the Espresso and UI Automator 2.0 testing\n            frameworks.\n          - *game-loop*: executes a special intent built into the game app (a\n            "demo mode") that simulates the actions of a real player. This test\n            type can include multiple game loops (also called "scenarios"),\n            which can be logically organized using scenario labels so that you\n            can run related loops together. Refer to\n            https://firebase.google.com/docs/test-lab/android/game-loop for\n            more information about how to build and run Game Loop tests.\n\n          The type of test to run can be specified with the *--type* flag,\n          although the type can often be inferred from other flags.\n          Specifically, if the *--test* flag is present, the test *--type*\n          defaults to `instrumentation`. If *--test* is not present, then\n          *--type* defaults to `robo`.\n\n          All arguments for *{command}* may be specified on the command line\n          and/or within an argument file. Run *$ gcloud topic arg-files* for\n          more information about argument files.\n          ', 'EXAMPLES': '\n          To invoke a robo test lasting 100 seconds against the default device\n          environment, run:\n\n            $ {command} --app=APP_APK --timeout=100s\n\n          When specifying devices to test against, the preferred method is to\n          use the --device flag. For example, to invoke a robo test against a\n          virtual, generic MDPI Nexus device in landscape orientation, run:\n\n            $ {command} --app=APP_APK --device=model=NexusLowRes,orientation=landscape\n\n          To invoke an instrumentation test against a physical Nexus 6 device\n          (MODEL_ID: shamu) which is running Android API level 21 in French, run:\n\n            $ {command} --app=APP_APK --test=TEST_APK --device=model=shamu,version=21,locale=fr\n\n          To test against multiple devices, specify --device more than once:\n\n            $ {command} --app=APP_APK --test=TEST_APK --device=model=Nexus4,version=19 --device=model=Nexus4,version=21 --device=model=NexusLowRes,version=25\n\n          To invoke a robo test on an Android App Bundle, pass the .aab file\n          using the --app flag.\n\n            $ {command} --app=bundle.aab\n\n          You may also use the legacy dimension flags (deprecated) to specify\n          which devices to use. Firebase Test Lab will run tests against every\n          possible combination of the listed device dimensions. Note that some\n          combinations of device models and OS versions may not be valid or\n          available in Test Lab. Any unsupported combinations of dimensions in\n          the test matrix will be skipped.\n\n          For example, to execute a series of 5-minute robo tests against a very\n          comprehensive matrix of virtual and physical devices, OS versions,\n          locales and orientations, run:\n\n            $ {command} --app=APP_APK --timeout=5m --device-ids=shamu,NexusLowRes,Nexus5,g3,zeroflte --os-version-ids=19,21,22,23,24,25 --locales=en_GB,es,fr,ru,zh --orientations=portrait,landscape\n\n          The above command will generate a test matrix with a total of 300 test\n          executions, but only the subset of executions with valid dimension\n          combinations will actually run your tests.\n\n          To help you identify and locate your test matrix in the Firebase\n          console, run:\n\n            $ {command} --app=APP_APK --client-details=matrixLabel="Example matrix label"\n\n          Controlling Results Storage\n\n          By default, Firebase Test Lab stores detailed test results for a\n          limited time in a Google Cloud Storage bucket provided for you at\n          no charge. If you wish to use a storage bucket that you control, or\n          if you need to retain detailed test results for a longer period,\n          use the *--results-bucket* option. See\n          https://firebase.google.com/docs/test-lab/analyzing-results#detailed\n          for more information.\n\n          Detailed test result files are prefixed by default with a timestamp\n          and a random character string. If you require a predictable path\n          where detailed test results are stored within the results bucket\n          (say, if you have a Continuous Integration system which does custom\n          post-processing of test result artifacts), use the *--results-dir*\n          option. _Note that each test invocation *must* have a unique storage\n          location, so never reuse the same value for *--results-dir* between\n          different test runs_. Possible strategies could include using a UUID\n          or sequence number for *--results-dir*.\n\n          For example, to run a robo test using a specific Google Cloud Storage\n          location to hold the raw test results, run:\n\n            $ {command} --app=APP_APK --results-bucket=gs://my-bucket --results-dir=my/test/results/<unique-value>\n\n          To run an instrumentation test and specify a custom name under which\n          the history of your tests will be collected and displayed in the\n          Firebase console, run:\n\n            $ {command} --app=APP_APK --test=TEST_APK --results-history-name=\'Excelsior App Test History\'\n\n          Argument Files\n\n          All test arguments for a given test may alternatively be stored in an\n          argument group within a YAML-formatted argument file. The _ARG_FILE_\n          may contain one or more named argument groups, and argument groups may\n          be combined using the `include:` attribute (Run *$ gcloud topic\n          arg-files* for more information). The ARG_FILE can easily be shared\n          with colleagues or placed under source control to ensure consistent\n          test executions.\n\n          To run a test using arguments loaded from an ARG_FILE named\n          *excelsior_args*, which contains an argument group named *robo-args:*,\n          use the following syntax:\n\n            $ {command} path/to/excelsior_args:robo-args\n          '}

    def Run(self, args):
        """Run the 'gcloud firebase test run' command to invoke a test in Test Lab.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation (i.e. group and command arguments combined).

    Returns:
      One of:
        - a list of TestOutcome tuples (if ToolResults are available).
        - a URL string pointing to the user's results in ToolResults or GCS.
    """
        if args.async_ and (not args.IsSpecified('format')):
            args.format = "\n          value(format(\n            'Final test results will be available at [ {0} ].', [])\n          )\n      "
        log.status.Print('\nHave questions, feedback, or issues? Get support by visiting:\n  https://firebase.google.com/support/\n')
        arg_manager.AndroidArgsManager().Prepare(args)
        project = util.GetProject()
        tr_client = self.context['toolresults_client']
        tr_messages = self.context['toolresults_messages']
        storage_client = self.context['storage_client']
        bucket_ops = results_bucket.ResultsBucketOps(project, args.results_bucket, args.results_dir, tr_client, tr_messages, storage_client)
        bucket_ops.UploadFileToGcs(args.app, _APK_MIME_TYPE)
        if args.test:
            bucket_ops.UploadFileToGcs(args.test, _APK_MIME_TYPE)
        for obb_file in args.obb_files or []:
            bucket_ops.UploadFileToGcs(obb_file, 'application/octet-stream')
        if getattr(args, 'robo_script', None):
            bucket_ops.UploadFileToGcs(args.robo_script, 'application/json')
        additional_apks = getattr(args, 'additional_apks', None) or []
        for additional_apk in additional_apks:
            bucket_ops.UploadFileToGcs(additional_apk, _APK_MIME_TYPE)
        other_files = getattr(args, 'other_files', None) or {}
        for device_path, file_to_upload in six.iteritems(other_files):
            bucket_ops.UploadFileToGcs(file_to_upload, None, destination_object=util.GetRelativeDevicePath(device_path))
        bucket_ops.LogGcsResultsUrl()
        tr_history_picker = history_picker.ToolResultsHistoryPicker(project, tr_client, tr_messages)
        history_name = PickHistoryName(args)
        history_id = tr_history_picker.GetToolResultsHistoryId(history_name)
        matrix = matrix_creator.CreateMatrix(args, self.context, history_id, bucket_ops.gcs_results_root, six.text_type(self.ReleaseTrack()))
        monitor = matrix_ops.MatrixMonitor(matrix.testMatrixId, args.type, self.context)
        with ctrl_c_handler.CancellableTestSection(monitor):
            tr_ids = tool_results.GetToolResultsIds(matrix, monitor)
            matrix = monitor.GetTestMatrixStatus()
            supported_executions = monitor.HandleUnsupportedExecutions(matrix)
            url = tool_results.CreateToolResultsUiUrl(project, tr_ids)
            log.status.Print('')
            if args.async_:
                return url
            log.status.Print('Test results will be streamed to [ {0} ].'.format(url))
            if len(supported_executions) == 1 and args.num_flaky_test_attempts == 0:
                monitor.MonitorTestExecutionProgress(supported_executions[0].id)
            else:
                monitor.MonitorTestMatrixProgress()
        log.status.Print('\nMore details are available at [ {0} ].'.format(url))
        summary_fetcher = results_summary.ToolResultsSummaryFetcher(project, tr_client, tr_messages, tr_ids, matrix.testMatrixId)
        self.exit_code = exit_code.ExitCodeFromRollupOutcome(summary_fetcher.FetchMatrixRollupOutcome(), tr_messages.Outcome.SummaryValueValuesEnum)
        return summary_fetcher.CreateMatrixOutcomeSummaryUsingEnvironments()