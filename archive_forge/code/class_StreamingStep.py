from boto.compat import six
class StreamingStep(Step):
    """
    Hadoop streaming step
    """

    def __init__(self, name, mapper, reducer=None, combiner=None, action_on_failure='TERMINATE_JOB_FLOW', cache_files=None, cache_archives=None, step_args=None, input=None, output=None, jar='/home/hadoop/contrib/streaming/hadoop-streaming.jar'):
        """
        A hadoop streaming elastic mapreduce step

        :type name: str
        :param name: The name of the step
        :type mapper: str
        :param mapper: The mapper URI
        :type reducer: str
        :param reducer: The reducer URI
        :type combiner: str
        :param combiner: The combiner URI. Only works for Hadoop 0.20
            and later!
        :type action_on_failure: str
        :param action_on_failure: An action, defined in the EMR docs to
            take on failure.
        :type cache_files: list(str)
        :param cache_files: A list of cache files to be bundled with the job
        :type cache_archives: list(str)
        :param cache_archives: A list of jar archives to be bundled with
            the job
        :type step_args: list(str)
        :param step_args: A list of arguments to pass to the step
        :type input: str or a list of str
        :param input: The input uri
        :type output: str
        :param output: The output uri
        :type jar: str
        :param jar: The hadoop streaming jar. This can be either a local
            path on the master node, or an s3:// URI.
        """
        self.name = name
        self.mapper = mapper
        self.reducer = reducer
        self.combiner = combiner
        self.action_on_failure = action_on_failure
        self.cache_files = cache_files
        self.cache_archives = cache_archives
        self.input = input
        self.output = output
        self._jar = jar
        if isinstance(step_args, six.string_types):
            step_args = [step_args]
        self.step_args = step_args

    def jar(self):
        return self._jar

    def main_class(self):
        return None

    def args(self):
        args = []
        if self.step_args:
            args.extend(self.step_args)
        args.extend(['-mapper', self.mapper])
        if self.combiner:
            args.extend(['-combiner', self.combiner])
        if self.reducer:
            args.extend(['-reducer', self.reducer])
        else:
            args.extend(['-jobconf', 'mapred.reduce.tasks=0'])
        if self.input:
            if isinstance(self.input, list):
                for input in self.input:
                    args.extend(('-input', input))
            else:
                args.extend(('-input', self.input))
        if self.output:
            args.extend(('-output', self.output))
        if self.cache_files:
            for cache_file in self.cache_files:
                args.extend(('-cacheFile', cache_file))
        if self.cache_archives:
            for cache_archive in self.cache_archives:
                args.extend(('-cacheArchive', cache_archive))
        return args

    def __repr__(self):
        return '%s.%s(name=%r, mapper=%r, reducer=%r, action_on_failure=%r, cache_files=%r, cache_archives=%r, step_args=%r, input=%r, output=%r, jar=%r)' % (self.__class__.__module__, self.__class__.__name__, self.name, self.mapper, self.reducer, self.action_on_failure, self.cache_files, self.cache_archives, self.step_args, self.input, self.output, self._jar)