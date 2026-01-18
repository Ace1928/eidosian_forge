import logging
import sys
import warnings
class LogConfigurator(object):
    _CONSOLE_MESSAGE_FORMAT = '%(message)s'

    def __init__(self, options):
        self.root_logger = logging.getLogger('')
        self.root_logger.setLevel(logging.DEBUG)
        self.dump_trace = False
        if options.debug:
            options.verbose_level = 3
            self.dump_trace = True
        self.console_logger = logging.StreamHandler(sys.stderr)
        log_level = log_level_from_options(options)
        self.console_logger.setLevel(log_level)
        formatter = logging.Formatter(self._CONSOLE_MESSAGE_FORMAT)
        self.console_logger.setFormatter(formatter)
        self.root_logger.addHandler(self.console_logger)
        set_warning_filter(log_level)
        self.file_logger = None
        log_file = options.log_file
        if log_file:
            self.file_logger = logging.FileHandler(filename=log_file)
            self.file_logger.setFormatter(_FileFormatter(options=options))
            self.file_logger.setLevel(log_level)
            self.root_logger.addHandler(self.file_logger)
        requests_log = logging.getLogger('requests')
        cliff_log = logging.getLogger('cliff')
        stevedore_log = logging.getLogger('stevedore')
        iso8601_log = logging.getLogger('iso8601')
        if options.debug:
            requests_log.setLevel(logging.DEBUG)
        else:
            requests_log.setLevel(logging.ERROR)
        cliff_log.setLevel(logging.ERROR)
        stevedore_log.setLevel(logging.ERROR)
        iso8601_log.setLevel(logging.ERROR)

    def configure(self, cloud_config):
        log_level = log_level_from_config(cloud_config.config)
        set_warning_filter(log_level)
        self.dump_trace = cloud_config.config.get('debug', self.dump_trace)
        self.console_logger.setLevel(log_level)
        log_file = cloud_config.config.get('log_file')
        if log_file:
            if not self.file_logger:
                self.file_logger = logging.FileHandler(filename=log_file)
            self.file_logger.setFormatter(_FileFormatter(config=cloud_config))
            self.file_logger.setLevel(log_level)
            self.root_logger.addHandler(self.file_logger)
        logconfig = cloud_config.config.get('logging')
        if logconfig:
            highest_level = logging.NOTSET
            for k in logconfig.keys():
                level = log_level_from_string(logconfig[k])
                logging.getLogger(k).setLevel(level)
                if highest_level < level:
                    highest_level = level
            self.console_logger.setLevel(highest_level)
            if self.file_logger:
                self.file_logger.setLevel(highest_level)
            for logkey in logging.Logger.manager.loggerDict.keys():
                logger = logging.getLogger(logkey)
                if logger.level == logging.NOTSET:
                    logger.setLevel(log_level)