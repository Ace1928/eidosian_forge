from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataProfileResultProfileFieldProfileInfoDoubleFieldInfo(_messages.Message):
    """The profile information for a double type field.

  Fields:
    average: Average of non-null values in the scanned data. NaN, if the field
      has a NaN.
    max: Maximum of non-null values in the scanned data. NaN, if the field has
      a NaN.
    min: Minimum of non-null values in the scanned data. NaN, if the field has
      a NaN.
    quartiles: A quartile divides the number of data points into four parts,
      or quarters, of more-or-less equal size. Three main quartiles used are:
      The first quartile (Q1) splits off the lowest 25% of data from the
      highest 75%. It is also known as the lower or 25th empirical quartile,
      as 25% of the data is below this point. The second quartile (Q2) is the
      median of a data set. So, 50% of the data lies below this point. The
      third quartile (Q3) splits off the highest 25% of data from the lowest
      75%. It is known as the upper or 75th empirical quartile, as 75% of the
      data lies below this point. Here, the quartiles is provided as an
      ordered list of quartile values for the scanned data, occurring in order
      Q1, median, Q3.
    standardDeviation: Standard deviation of non-null values in the scanned
      data. NaN, if the field has a NaN.
  """
    average = _messages.FloatField(1)
    max = _messages.FloatField(2)
    min = _messages.FloatField(3)
    quartiles = _messages.FloatField(4, repeated=True)
    standardDeviation = _messages.FloatField(5)