from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AirflowDag(_messages.Message):
    """A specification of an Apache Airflow DAG.

  Fields:
    data: The contents of a Python module or ZIP archive that specifies an
      Airflow DAG.
  """
    data = _messages.BytesField(1)