import warnings
import sys
import socket
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..module import get_introspection_module
from .._gi import (variant_type_from_string, source_new,
from ..overrides import override, deprecated, deprecated_attr
from gi import PyGIDeprecationWarning, version_info
from gi import _option as option
from gi import _gi
from gi._error import GError
class _VariantCreator(object):
    _LEAF_CONSTRUCTORS = {'b': GLib.Variant.new_boolean, 'y': GLib.Variant.new_byte, 'n': GLib.Variant.new_int16, 'q': GLib.Variant.new_uint16, 'i': GLib.Variant.new_int32, 'u': GLib.Variant.new_uint32, 'x': GLib.Variant.new_int64, 't': GLib.Variant.new_uint64, 'h': GLib.Variant.new_handle, 'd': GLib.Variant.new_double, 's': GLib.Variant.new_string, 'o': GLib.Variant.new_object_path, 'g': GLib.Variant.new_signature, 'v': GLib.Variant.new_variant}

    def _create(self, format, value):
        """Create a GVariant object from given format and a value that matches
        the format.

        This method recursively calls itself for complex structures (arrays,
        dictionaries, boxed).

        Returns the generated GVariant.

        If value is None it will generate an empty GVariant container type.
        """
        gvtype = GLib.VariantType(format)
        if format in self._LEAF_CONSTRUCTORS:
            return self._LEAF_CONSTRUCTORS[format](value)
        builder = GLib.VariantBuilder.new(gvtype)
        if value is None:
            return builder.end()
        if gvtype.is_maybe():
            builder.add_value(self._create(gvtype.element().dup_string(), value))
            return builder.end()
        try:
            iter(value)
        except TypeError:
            raise TypeError('Could not create array, tuple or dictionary entry from non iterable value %s %s' % (format, value))
        if gvtype.is_tuple() and gvtype.n_items() != len(value):
            raise TypeError("Tuple mismatches value's number of elements %s %s" % (format, value))
        if gvtype.is_dict_entry() and len(value) != 2:
            raise TypeError('Dictionary entries must have two elements %s %s' % (format, value))
        if gvtype.is_array():
            element_type = gvtype.element().dup_string()
            if isinstance(value, dict):
                value = value.items()
            for i in value:
                builder.add_value(self._create(element_type, i))
        else:
            remainer_format = format[1:]
            for i in value:
                dup = variant_type_from_string(remainer_format).dup_string()
                builder.add_value(self._create(dup, i))
                remainer_format = remainer_format[len(dup):]
        return builder.end()