import re
import types
import sys
import os.path
import inspect
import base64
import warnings
class LRTable(object):

    def __init__(self):
        self.lr_action = None
        self.lr_goto = None
        self.lr_productions = None
        self.lr_method = None

    def read_table(self, module):
        if isinstance(module, types.ModuleType):
            parsetab = module
        else:
            exec('import %s' % module)
            parsetab = sys.modules[module]
        if parsetab._tabversion != __tabversion__:
            raise VersionError('yacc table file version is out of date')
        self.lr_action = parsetab._lr_action
        self.lr_goto = parsetab._lr_goto
        self.lr_productions = []
        for p in parsetab._lr_productions:
            self.lr_productions.append(MiniProduction(*p))
        self.lr_method = parsetab._lr_method
        return parsetab._lr_signature

    def read_pickle(self, filename):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        if not os.path.exists(filename):
            raise ImportError
        in_f = open(filename, 'rb')
        tabversion = pickle.load(in_f)
        if tabversion != __tabversion__:
            raise VersionError('yacc table file version is out of date')
        self.lr_method = pickle.load(in_f)
        signature = pickle.load(in_f)
        self.lr_action = pickle.load(in_f)
        self.lr_goto = pickle.load(in_f)
        productions = pickle.load(in_f)
        self.lr_productions = []
        for p in productions:
            self.lr_productions.append(MiniProduction(*p))
        in_f.close()
        return signature

    def bind_callables(self, pdict):
        for p in self.lr_productions:
            p.bind(pdict)