from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OptionalComponentsValueListEntryValuesEnum(_messages.Enum):
    """OptionalComponentsValueListEntryValuesEnum enum type.

    Values:
      COMPONENT_UNSPECIFIED: Unspecified component. Specifying this will cause
        Cluster creation to fail.
      ANACONDA: The Anaconda python distribution. The Anaconda component is
        not supported in the Dataproc 2.0 image. The 2.0 image is pre-
        installed with Miniconda.
      DOCKER: Docker
      DRUID: The Druid query engine. (alpha)
      FLINK: Flink
      HBASE: HBase. (beta)
      HIVE_WEBHCAT: The Hive Web HCatalog (the REST service for accessing
        HCatalog).
      HUDI: Hudi.
      JUPYTER: The Jupyter Notebook.
      KERBEROS: The Kerberos security feature.
      PRESTO: The Presto query engine.
      TRINO: The Trino query engine.
      RANGER: The Ranger service.
      SOLR: The Solr service.
      ZEPPELIN: The Zeppelin notebook.
      ZOOKEEPER: The Zookeeper service.
      DASK: Dask
      GPU_DRIVER: Nvidia GPU driver.
    """
    COMPONENT_UNSPECIFIED = 0
    ANACONDA = 1
    DOCKER = 2
    DRUID = 3
    FLINK = 4
    HBASE = 5
    HIVE_WEBHCAT = 6
    HUDI = 7
    JUPYTER = 8
    KERBEROS = 9
    PRESTO = 10
    TRINO = 11
    RANGER = 12
    SOLR = 13
    ZEPPELIN = 14
    ZOOKEEPER = 15
    DASK = 16
    GPU_DRIVER = 17