import os
from apache_beam.io.filesystems import FileSystems
from apache_beam.pipeline import Pipeline
from .logging import get_logger
Use the Beam Filesystems to download from a remote directory on gcs/s3/hdfs...