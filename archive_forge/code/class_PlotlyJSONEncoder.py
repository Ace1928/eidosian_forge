import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
class PlotlyJSONEncoder(_json.JSONEncoder):
    """
    Meant to be passed as the `cls` kwarg to json.dumps(obj, cls=..)

    See PlotlyJSONEncoder.default for more implementation information.

    Additionally, this encoder overrides nan functionality so that 'Inf',
    'NaN' and '-Inf' encode to 'null'. Which is stricter JSON than the Python
    version.

    """

    def coerce_to_strict(self, const):
        """
        This is used to ultimately *encode* into strict JSON, see `encode`

        """
        if const in ('Infinity', '-Infinity', 'NaN'):
            return None
        else:
            return const

    def encode(self, o):
        """
        Load and then dump the result using parse_constant kwarg

        Note that setting invalid separators will cause a failure at this step.

        """
        encoded_o = super(PlotlyJSONEncoder, self).encode(o)
        if not ('NaN' in encoded_o or 'Infinity' in encoded_o):
            return encoded_o
        try:
            new_o = _json.loads(encoded_o, parse_constant=self.coerce_to_strict)
        except ValueError:
            raise ValueError('Encoding into strict JSON failed. Did you set the separators valid JSON separators?')
        else:
            return _json.dumps(new_o, sort_keys=self.sort_keys, indent=self.indent, separators=(self.item_separator, self.key_separator))

    def default(self, obj):
        """
        Accept an object (of unknown type) and try to encode with priority:
        1. builtin:     user-defined objects
        2. sage:        sage math cloud
        3. pandas:      dataframes/series
        4. numpy:       ndarrays
        5. datetime:    time/datetime objects

        Each method throws a NotEncoded exception if it fails.

        The default method will only get hit if the object is not a type that
        is naturally encoded by json:

            Normal objects:
                dict                object
                list, tuple         array
                str, unicode        string
                int, long, float    number
                True                true
                False               false
                None                null

            Extended objects:
                float('nan')        'NaN'
                float('infinity')   'Infinity'
                float('-infinity')  '-Infinity'

        Therefore, we only anticipate either unknown iterables or values here.

        """
        encoding_methods = (self.encode_as_plotly, self.encode_as_sage, self.encode_as_numpy, self.encode_as_pandas, self.encode_as_datetime, self.encode_as_date, self.encode_as_list, self.encode_as_decimal, self.encode_as_pil)
        for encoding_method in encoding_methods:
            try:
                return encoding_method(obj)
            except NotEncodable:
                pass
        return _json.JSONEncoder.default(self, obj)

    @staticmethod
    def encode_as_plotly(obj):
        """Attempt to use a builtin `to_plotly_json` method."""
        try:
            return obj.to_plotly_json()
        except AttributeError:
            raise NotEncodable

    @staticmethod
    def encode_as_list(obj):
        """Attempt to use `tolist` method to convert to normal Python list."""
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            raise NotEncodable

    @staticmethod
    def encode_as_sage(obj):
        """Attempt to convert sage.all.RR to floats and sage.all.ZZ to ints"""
        sage_all = get_module('sage.all')
        if not sage_all:
            raise NotEncodable
        if obj in sage_all.RR:
            return float(obj)
        elif obj in sage_all.ZZ:
            return int(obj)
        else:
            raise NotEncodable

    @staticmethod
    def encode_as_pandas(obj):
        """Attempt to convert pandas.NaT / pandas.NA"""
        pandas = get_module('pandas', should_load=False)
        if not pandas:
            raise NotEncodable
        if obj is pandas.NaT:
            return None
        if hasattr(pandas, 'NA') and obj is pandas.NA:
            return None
        raise NotEncodable

    @staticmethod
    def encode_as_numpy(obj):
        """Attempt to convert numpy.ma.core.masked"""
        numpy = get_module('numpy', should_load=False)
        if not numpy:
            raise NotEncodable
        if obj is numpy.ma.core.masked:
            return float('nan')
        elif isinstance(obj, numpy.ndarray) and obj.dtype.kind == 'M':
            try:
                return numpy.datetime_as_string(obj).tolist()
            except TypeError:
                pass
        raise NotEncodable

    @staticmethod
    def encode_as_datetime(obj):
        """Convert datetime objects to iso-format strings"""
        try:
            return obj.isoformat()
        except AttributeError:
            raise NotEncodable

    @staticmethod
    def encode_as_date(obj):
        """Attempt to convert to utc-iso time string using date methods."""
        try:
            time_string = obj.isoformat()
        except AttributeError:
            raise NotEncodable
        else:
            return iso_to_plotly_time_string(time_string)

    @staticmethod
    def encode_as_decimal(obj):
        """Attempt to encode decimal by converting it to float"""
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        else:
            raise NotEncodable

    @staticmethod
    def encode_as_pil(obj):
        """Attempt to convert PIL.Image.Image to base64 data uri"""
        image = get_module('PIL.Image')
        if image is not None and isinstance(obj, image.Image):
            return ImageUriValidator.pil_image_to_uri(obj)
        else:
            raise NotEncodable