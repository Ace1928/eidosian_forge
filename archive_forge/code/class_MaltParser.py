import inspect
import os
import subprocess
import sys
import tempfile
from nltk.data import ZipFilePathPointer
from nltk.internals import find_dir, find_file, find_jars_within_path
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.parse.util import taggedsents_to_conll
class MaltParser(ParserI):
    """
    A class for dependency parsing with MaltParser. The input is the paths to:
    - (optionally) a maltparser directory
    - (optionally) the path to a pre-trained MaltParser .mco model file
    - (optionally) the tagger to use for POS tagging before parsing
    - (optionally) additional Java arguments

    Example:
        >>> from nltk.parse import malt
        >>> # With MALT_PARSER and MALT_MODEL environment set.
        >>> mp = malt.MaltParser(model_filename='engmalt.linear-1.7.mco') # doctest: +SKIP
        >>> mp.parse_one('I shot an elephant in my pajamas .'.split()).tree() # doctest: +SKIP
        (shot I (elephant an) (in (pajamas my)) .)
        >>> # Without MALT_PARSER and MALT_MODEL environment.
        >>> mp = malt.MaltParser('/home/user/maltparser-1.9.2/', '/home/user/engmalt.linear-1.7.mco') # doctest: +SKIP
        >>> mp.parse_one('I shot an elephant in my pajamas .'.split()).tree() # doctest: +SKIP
        (shot I (elephant an) (in (pajamas my)) .)
    """

    def __init__(self, parser_dirname='', model_filename=None, tagger=None, additional_java_args=None):
        """
        An interface for parsing with the Malt Parser.

        :param parser_dirname: The path to the maltparser directory that
            contains the maltparser-1.x.jar
        :type parser_dirname: str
        :param model_filename: The name of the pre-trained model with .mco file
            extension. If provided, training will not be required.
            (see http://www.maltparser.org/mco/mco.html and
            see http://www.patful.com/chalk/node/185)
        :type model_filename: str
        :param tagger: The tagger used to POS tag the raw string before
            formatting to CONLL format. It should behave like `nltk.pos_tag`
        :type tagger: function
        :param additional_java_args: This is the additional Java arguments that
            one can use when calling Maltparser, usually this is the heapsize
            limits, e.g. `additional_java_args=['-Xmx1024m']`
            (see https://goo.gl/mpDBvQ)
        :type additional_java_args: list
        """
        self.malt_jars = find_maltparser(parser_dirname)
        self.additional_java_args = additional_java_args if additional_java_args is not None else []
        self.model = find_malt_model(model_filename)
        self._trained = self.model != 'malt_temp.mco'
        self.working_dir = tempfile.gettempdir()
        self.tagger = tagger if tagger is not None else malt_regex_tagger()

    def parse_tagged_sents(self, sentences, verbose=False, top_relation_label='null'):
        """
        Use MaltParser to parse multiple POS tagged sentences. Takes multiple
        sentences where each sentence is a list of (word, tag) tuples.
        The sentences must have already been tokenized and tagged.

        :param sentences: Input sentences to parse
        :type sentence: list(list(tuple(str, str)))
        :return: iter(iter(``DependencyGraph``)) the dependency graph
            representation of each sentence
        """
        if not self._trained:
            raise Exception('Parser has not been trained. Call train() first.')
        with tempfile.NamedTemporaryFile(prefix='malt_input.conll.', dir=self.working_dir, mode='w', delete=False) as input_file:
            with tempfile.NamedTemporaryFile(prefix='malt_output.conll.', dir=self.working_dir, mode='w', delete=False) as output_file:
                for line in taggedsents_to_conll(sentences):
                    input_file.write(str(line))
                input_file.close()
                cmd = self.generate_malt_command(input_file.name, output_file.name, mode='parse')
                _current_path = os.getcwd()
                try:
                    os.chdir(os.path.split(self.model)[0])
                except:
                    pass
                ret = self._execute(cmd, verbose)
                os.chdir(_current_path)
                if ret != 0:
                    raise Exception('MaltParser parsing (%s) failed with exit code %d' % (' '.join(cmd), ret))
                with open(output_file.name) as infile:
                    for tree_str in infile.read().split('\n\n'):
                        yield iter([DependencyGraph(tree_str, top_relation_label=top_relation_label)])
        os.remove(input_file.name)
        os.remove(output_file.name)

    def parse_sents(self, sentences, verbose=False, top_relation_label='null'):
        """
        Use MaltParser to parse multiple sentences.
        Takes a list of sentences, where each sentence is a list of words.
        Each sentence will be automatically tagged with this
        MaltParser instance's tagger.

        :param sentences: Input sentences to parse
        :type sentence: list(list(str))
        :return: iter(DependencyGraph)
        """
        tagged_sentences = (self.tagger(sentence) for sentence in sentences)
        return self.parse_tagged_sents(tagged_sentences, verbose, top_relation_label=top_relation_label)

    def generate_malt_command(self, inputfilename, outputfilename=None, mode=None):
        """
        This function generates the maltparser command use at the terminal.

        :param inputfilename: path to the input file
        :type inputfilename: str
        :param outputfilename: path to the output file
        :type outputfilename: str
        """
        cmd = ['java']
        cmd += self.additional_java_args
        classpaths_separator = ';' if sys.platform.startswith('win') else ':'
        cmd += ['-cp', classpaths_separator.join(self.malt_jars)]
        cmd += ['org.maltparser.Malt']
        if os.path.exists(self.model):
            cmd += ['-c', os.path.split(self.model)[-1]]
        else:
            cmd += ['-c', self.model]
        cmd += ['-i', inputfilename]
        if mode == 'parse':
            cmd += ['-o', outputfilename]
        cmd += ['-m', mode]
        return cmd

    @staticmethod
    def _execute(cmd, verbose=False):
        output = None if verbose else subprocess.PIPE
        p = subprocess.Popen(cmd, stdout=output, stderr=output)
        return p.wait()

    def train(self, depgraphs, verbose=False):
        """
        Train MaltParser from a list of ``DependencyGraph`` objects

        :param depgraphs: list of ``DependencyGraph`` objects for training input data
        :type depgraphs: DependencyGraph
        """
        with tempfile.NamedTemporaryFile(prefix='malt_train.conll.', dir=self.working_dir, mode='w', delete=False) as input_file:
            input_str = '\n'.join((dg.to_conll(10) for dg in depgraphs))
            input_file.write(str(input_str))
        self.train_from_file(input_file.name, verbose=verbose)
        os.remove(input_file.name)

    def train_from_file(self, conll_file, verbose=False):
        """
        Train MaltParser from a file
        :param conll_file: str for the filename of the training input data
        :type conll_file: str
        """
        if isinstance(conll_file, ZipFilePathPointer):
            with tempfile.NamedTemporaryFile(prefix='malt_train.conll.', dir=self.working_dir, mode='w', delete=False) as input_file:
                with conll_file.open() as conll_input_file:
                    conll_str = conll_input_file.read()
                    input_file.write(str(conll_str))
                return self.train_from_file(input_file.name, verbose=verbose)
        cmd = self.generate_malt_command(conll_file, mode='learn')
        ret = self._execute(cmd, verbose)
        if ret != 0:
            raise Exception('MaltParser training (%s) failed with exit code %d' % (' '.join(cmd), ret))
        self._trained = True