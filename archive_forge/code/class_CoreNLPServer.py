import json
import os  # required for doctests
import re
import socket
import time
from typing import List, Tuple
from nltk.internals import _java_options, config_java, find_jar_iter, java
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.tag.api import TaggerI
from nltk.tokenize.api import TokenizerI
from nltk.tree import Tree
class CoreNLPServer:
    _MODEL_JAR_PATTERN = 'stanford-corenlp-(\\d+)\\.(\\d+)\\.(\\d+)-models\\.jar'
    _JAR = 'stanford-corenlp-(\\d+)\\.(\\d+)\\.(\\d+)\\.jar'

    def __init__(self, path_to_jar=None, path_to_models_jar=None, verbose=False, java_options=None, corenlp_options=None, port=None):
        if corenlp_options is None:
            corenlp_options = ['-preload', 'tokenize,ssplit,pos,lemma,parse,depparse']
        jars = list(find_jar_iter(self._JAR, path_to_jar, env_vars=('CORENLP',), searchpath=(), url=_stanford_url, verbose=verbose, is_regex=True))
        stanford_jar = max(jars, key=lambda model_name: re.match(self._JAR, model_name))
        if port is None:
            try:
                port = try_port(9000)
            except OSError:
                port = try_port()
                corenlp_options.extend(['-port', str(port)])
        else:
            try_port(port)
            corenlp_options.extend(['-port', str(port)])
        self.url = f'http://localhost:{port}'
        model_jar = max(find_jar_iter(self._MODEL_JAR_PATTERN, path_to_models_jar, env_vars=('CORENLP_MODELS',), searchpath=(), url=_stanford_url, verbose=verbose, is_regex=True), key=lambda model_name: re.match(self._MODEL_JAR_PATTERN, model_name))
        self.verbose = verbose
        self._classpath = (stanford_jar, model_jar)
        self.corenlp_options = corenlp_options
        self.java_options = java_options or ['-mx2g']

    def start(self, stdout='devnull', stderr='devnull'):
        """Starts the CoreNLP server

        :param stdout, stderr: Specifies where CoreNLP output is redirected. Valid values are 'devnull', 'stdout', 'pipe'
        """
        import requests
        cmd = ['edu.stanford.nlp.pipeline.StanfordCoreNLPServer']
        if self.corenlp_options:
            cmd.extend(self.corenlp_options)
        default_options = ' '.join(_java_options)
        config_java(options=self.java_options, verbose=self.verbose)
        try:
            self.popen = java(cmd, classpath=self._classpath, blocking=False, stdout=stdout, stderr=stderr)
        finally:
            config_java(options=default_options, verbose=self.verbose)
        returncode = self.popen.poll()
        if returncode is not None:
            _, stderrdata = self.popen.communicate()
            raise CoreNLPServerError(returncode, 'Could not start the server. The error was: {}'.format(stderrdata.decode('ascii')))
        for i in range(30):
            try:
                response = requests.get(requests.compat.urljoin(self.url, 'live'))
            except requests.exceptions.ConnectionError:
                time.sleep(1)
            else:
                if response.ok:
                    break
        else:
            raise CoreNLPServerError('Could not connect to the server.')
        for i in range(60):
            try:
                response = requests.get(requests.compat.urljoin(self.url, 'ready'))
            except requests.exceptions.ConnectionError:
                time.sleep(1)
            else:
                if response.ok:
                    break
        else:
            raise CoreNLPServerError('The server is not ready.')

    def stop(self):
        self.popen.terminate()
        self.popen.wait()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False