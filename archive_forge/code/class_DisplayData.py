from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DisplayData(_messages.Message):
    """Data provided with a pipeline or transform to provide descriptive info.

  Fields:
    boolValue: Contains value if the data is of a boolean type.
    durationValue: Contains value if the data is of duration type.
    floatValue: Contains value if the data is of float type.
    int64Value: Contains value if the data is of int64 type.
    javaClassValue: Contains value if the data is of java class type.
    key: The key identifying the display data. This is intended to be used as
      a label for the display data when viewed in a dax monitoring system.
    label: An optional label to display in a dax UI for the element.
    namespace: The namespace for the key. This is usually a class name or
      programming language namespace (i.e. python module) which defines the
      display data. This allows a dax monitoring system to specially handle
      the data and perform custom rendering.
    shortStrValue: A possible additional shorter value to display. For example
      a java_class_name_value of com.mypackage.MyDoFn will be stored with
      MyDoFn as the short_str_value and com.mypackage.MyDoFn as the
      java_class_name value. short_str_value can be displayed and
      java_class_name_value will be displayed as a tooltip.
    strValue: Contains value if the data is of string type.
    timestampValue: Contains value if the data is of timestamp type.
    url: An optional full URL.
  """
    boolValue = _messages.BooleanField(1)
    durationValue = _messages.StringField(2)
    floatValue = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    int64Value = _messages.IntegerField(4)
    javaClassValue = _messages.StringField(5)
    key = _messages.StringField(6)
    label = _messages.StringField(7)
    namespace = _messages.StringField(8)
    shortStrValue = _messages.StringField(9)
    strValue = _messages.StringField(10)
    timestampValue = _messages.StringField(11)
    url = _messages.StringField(12)