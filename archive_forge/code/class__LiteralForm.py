import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
class _LiteralForm(_ExtensionsSpecialForm, _root=True):

    def __init__(self, doc: str):
        self._name = 'Literal'
        self._doc = self.__doc__ = doc

    def __getitem__(self, parameters):
        if not isinstance(parameters, tuple):
            parameters = (parameters,)
        parameters = _flatten_literal_params(parameters)
        val_type_pairs = list(_value_and_type_iter(parameters))
        try:
            deduped_pairs = set(val_type_pairs)
        except TypeError:
            pass
        else:
            if len(deduped_pairs) < len(val_type_pairs):
                new_parameters = []
                for pair in val_type_pairs:
                    if pair in deduped_pairs:
                        new_parameters.append(pair[0])
                        deduped_pairs.remove(pair)
                assert not deduped_pairs, deduped_pairs
                parameters = tuple(new_parameters)
        return _LiteralGenericAlias(self, parameters)