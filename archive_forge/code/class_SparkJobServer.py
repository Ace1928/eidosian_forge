import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from pyspark.util import inheritable_thread_target
from ray.util.spark.cluster_init import _start_ray_worker_nodes
class SparkJobServer(ThreadingHTTPServer):
    """
    High level design:

    1. In Ray on spark autoscaling mode, How to start and terminate Ray worker node ?

    It uses spark job to launch Ray worker node,
    and each spark job contains only one spark task, the corresponding spark task
    creates Ray worker node as subprocess.
    When autoscaler request terminating specific Ray worker node, it cancels
    corresponding spark job to trigger Ray worker node termination.
    Because we can only cancel spark job not spark task when we need to scale
    down a Ray worker node. So we have to have one spark job for each Ray worker node.

    2. How to create / cancel spark job from spark node provider?

    Spark node provider runs in autoscaler process that is different process
    than the one that executes "setup_ray_cluster" API. User calls "setup_ray_cluster"
    API in spark application driver node, and the semantic is "setup_ray_cluster"
    requests spark resources from this spark application.
    Internally, "setup_ray_cluster" should use "spark session" instance to request
    spark application resources. But spark node provider runs in another python
    process, in order to share spark session to the separate NodeProvider process,
    it sets up a spark job server that runs inside spark application driver process
    (the process that calls "setup_ray_cluster" API), and in NodeProvider process,
    it sends RPC request to the spark job server for creating spark jobs in the
    spark application.
    Note that we cannot create another spark session in NodeProvider process,
    because if doing so, it means we create another spark application, and then
    it causes NodeProvider requests resources belonging to the new spark application,
    but we need to ensure all requested spark resources belong to
    the original spark application that calls "setup_ray_cluster" API.

    Note:
    The server must inherit ThreadingHTTPServer because request handler uses
    the active spark session in current process to create spark jobs, so all request
    handler must be running in current process.
    """

    def __init__(self, server_address, spark):
        super().__init__(server_address, SparkJobServerRequestHandler)
        self.spark = spark
        self.task_status_dict = {}

    def shutdown(self) -> None:
        super().shutdown()
        for spark_job_group_id in self.task_status_dict:
            self.spark.sparkContext.cancelJobGroup(spark_job_group_id)