from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class AddJob(base.Group):
    """Add Dataproc jobs to workflow template.

  ## EXAMPLES

  To add a Hadoop MapReduce job, run:

    $ {command} hadoop --workflow-template my_template --jar my_jar.jar \\
        -- arg1 arg2

  To add a Spark Scala or Java job, run:

    $ {command} spark --workflow-template my_template --jar my_jar.jar \\
        -- arg1 arg2

  To add a PySpark job, run:

    $ {command} pyspark --workflow-template my_template my_script.py \\
        -- arg1 arg2

  To add a Spark SQL job, run:

    $ {command} spark-sql --workflow-template my_template --file my_queries.q

  To add a Pig job, run:

    $ {command} pig --workflow-template my_template --file my_script.pig

  To add a Hive job, run:

    $ {command} hive --workflow-template my_template --file my_queries.q
  """