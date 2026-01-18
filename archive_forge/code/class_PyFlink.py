from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc.jobs import pyflink
from googlecloudsdk.command_lib.dataproc.jobs import submitter
class PyFlink(pyflink.PyFlinkBase, submitter.JobSubmitter):
    """Submit a PyFlink job to a cluster.

  Submit a PyFlink job to a cluster.

  ## EXAMPLES

    Submit a PyFlink job.

    $ gcloud dataproc jobs submit pyflink my-pyflink.py --region=us-central1

    Submit a PyFlink job with additional source and resource files.

    $ gcloud dataproc jobs submit pyflink my-pyflink.py \\
      --region=us-central1 \\
      --py-files=my-python-file1.py,my-python-file2.py

    Submit a PyFlink job with a jar file.

    $ gcloud dataproc jobs submit pyflink my-pyflink.py \\
      --region=us-central1 \\
      --jars=my-jar-file.jar

    Submit a PyFlink job with 'python-files' and 'python-module'.

    $ gcloud dataproc jobs submit pyflink my-pyflink.py \\
      --region=us-central1 \\
      --py-files=my-python-file1.py,my-python-file2.py
      --py-module=my-module

  """

    @staticmethod
    def Args(parser):
        pyflink.PyFlinkBase.Args(parser)
        submitter.JobSubmitter.Args(parser)

    def ConfigureJob(self, messages, job, args):
        pyflink.PyFlinkBase.ConfigureJob(messages, job, self.files_by_type, self.BuildLoggingConfig(messages, args.driver_log_levels), args)
        submitter.JobSubmitter.ConfigureJob(messages, job, args)