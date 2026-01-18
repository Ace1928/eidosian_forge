from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
class SentencePieceNormalizer(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self):
        _sentencepiece.SentencePieceNormalizer_swiginit(self, _sentencepiece.new_SentencePieceNormalizer())
    __swig_destroy__ = _sentencepiece.delete_SentencePieceNormalizer

    def LoadFromSerializedProto(self, serialized):
        return _sentencepiece.SentencePieceNormalizer_LoadFromSerializedProto(self, serialized)

    def LoadFromRuleTSV(self, filename):
        return _sentencepiece.SentencePieceNormalizer_LoadFromRuleTSV(self, filename)

    def LoadFromRuleName(self, name):
        return _sentencepiece.SentencePieceNormalizer_LoadFromRuleName(self, name)

    def serialized_model_proto(self):
        return _sentencepiece.SentencePieceNormalizer_serialized_model_proto(self)

    def LoadFromFile(self, arg):
        return _sentencepiece.SentencePieceNormalizer_LoadFromFile(self, arg)

    def _Normalize(self, text):
        return _sentencepiece.SentencePieceNormalizer__Normalize(self, text)

    def _NormalizeWithOffsets(self, text):
        return _sentencepiece.SentencePieceNormalizer__NormalizeWithOffsets(self, text)

    def _SetProtoField(self, name, value):
        return _sentencepiece.SentencePieceNormalizer__SetProtoField(self, name, value)

    def Init(self, model_file=None, model_proto=None, rule_tsv=None, rule_name=None, add_dummy_prefix=False, escape_whitespaces=False, remove_extra_whitespaces=False):
        """Initialzie sentencePieceNormalizer.

      Args:
        model_file: The sentencepiece model file path.
        model_proto: The sentencepiece model serialized proto.
        rule_tsv: The normalization rule file in TSV format.
        rule_name: Pre-defined normalization name.
        add_dummy_prefix: add dummy prefix.
        escape_whitespaces: escape whitespaces.
        remove_extra_whitespaces: remove extra whitespaces.
      """
        _sentencepiece_normalizer_init_native(self)
        if model_file:
            status = self.LoadFromFile(model_file)
        elif model_proto:
            status = self.LoadFromSerializedProto(model_proto)
        elif rule_tsv:
            status = self.LoadFromRuleTSV(rule_tsv)
        elif rule_name:
            status = self.LoadFromRuleName(rule_name)
        else:
            raise RuntimeError('no model is specified')
        if status:
            self._SetProtoField('add_dummy_prefix', add_dummy_prefix)
            self._SetProtoField('escape_whitespaces', escape_whitespaces)
            self._SetProtoField('remove_extra_whitespaces', remove_extra_whitespaces)

    def Normalize(self, input, with_offsets=None):

        def _normalize(text):
            if with_offsets:
                return self._NormalizeWithOffsets(text)
            return self._Normalize(text)
        if type(input) is list:
            return [_normalize(x) for x in input]
        return _normalize(input)

    def __getstate__(self):
        return self.serialized_model_proto()

    def __setstate__(self, serialized_model_proto):
        self.__init__()
        self.LoadFromSerializedProto(serialized_model_proto)