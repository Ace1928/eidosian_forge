import os
import sys
from io import BytesIO
from types import CodeType, FunctionType
import dill
from packaging import version
from .. import config
def _save_spacyLanguage(pickler, obj):
    import spacy

    def create_spacyLanguage(config, bytes):
        lang_cls = spacy.util.get_lang_class(config['nlp']['lang'])
        lang_inst = lang_cls.from_config(config)
        return lang_inst.from_bytes(bytes)
    log(pickler, f'Sp: {obj}')
    args = (obj.config, obj.to_bytes())
    pickler.save_reduce(create_spacyLanguage, args, obj=obj)
    log(pickler, '# Sp')