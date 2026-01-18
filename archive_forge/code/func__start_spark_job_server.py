import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from pyspark.util import inheritable_thread_target
from ray.util.spark.cluster_init import _start_ray_worker_nodes
def _start_spark_job_server(host, port, spark):
    server = SparkJobServer((host, port), spark)

    def run_server():
        server.serve_forever()
    server_thread = threading.Thread(target=run_server)
    server_thread.setDaemon(True)
    server_thread.start()
    return server