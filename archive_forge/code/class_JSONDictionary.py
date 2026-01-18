import os.path
import json
from pyomo.common.collections import Bunch
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.dataportal.factory import DataManagerFactory
@DataManagerFactory.register('json', 'JSON file interface')
class JSONDictionary(object):

    def __init__(self):
        self._info = {}
        self.options = Bunch()

    def available(self):
        return True

    def initialize(self, **kwds):
        self.filename = kwds.pop('filename')
        self.add_options(**kwds)

    def add_options(self, **kwds):
        self.options.update(kwds)

    def open(self):
        if self.filename is None:
            raise IOError('No filename specified')

    def close(self):
        pass

    def read(self):
        """
        This function loads data from a JSON file and tuplizes the nested
        dictionaries and lists of lists.
        """
        if not os.path.exists(self.filename):
            raise IOError("Cannot find file '%s'" % self.filename)
        INPUT = open(self.filename, 'r')
        jdata = json.load(INPUT)
        INPUT.close()
        if jdata is None or len(jdata) == 0:
            raise IOError('Empty JSON data file')
        self._info = {}
        for k, v in jdata.items():
            self._info[k] = tuplize(v)

    def write(self, data):
        """
        This function creates a JSON file for the specified data.
        """
        with open(self.filename, 'w') as OUTPUT:
            jdata = {}
            if self.options.data is None:
                for k, v in data.items():
                    jdata[k] = detuplize(v)
            elif type(self.options.data) in (list, tuple):
                for k in self.options.data:
                    jdata[k] = detuplize(data[k], sort=self.options.sort)
            else:
                k = self.options.data
                jdata[k] = detuplize(data[k])
            json.dump(jdata, OUTPUT)

    def process(self, model, data, default):
        """
        Set the data for the selected components
        """
        if not self.options.namespace in data:
            data[self.options.namespace] = {}
        try:
            if self.options.data is None:
                for key in self._info:
                    self._set_data(data, self.options.namespace, key, self._info[key])
            elif type(self.options.data) in (list, tuple):
                for key in self.options.data:
                    self._set_data(data, self.options.namespace, key, self._info[key])
            else:
                key = self.options.data
                self._set_data(data, self.options.namespace, key, self._info[key])
        except KeyError:
            raise IOError("Data value for '%s' is not available in JSON file '%s'" % (key, self.filename))

    def _set_data(self, data, namespace, name, value):
        if type(value) is dict:
            data[namespace][name] = value
        else:
            data[namespace][name] = {None: value}

    def clear(self):
        self._info = {}