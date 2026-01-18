import os
from sphinx.util import logging
from oslo_config import generator
def _generate_sample(app, config_file, base_name):

    def info(msg):
        LOG.info('[%s] %s' % (__name__, msg))
    candidates = [config_file, os.path.join(app.srcdir, config_file)]
    for c in candidates:
        if os.path.isfile(c):
            info('reading config generator instructions from %s' % c)
            config_path = c
            break
    else:
        raise ValueError('Could not find config_generator_config_file %r' % app.config.config_generator_config_file)
    if base_name:
        out_file = os.path.join(app.srcdir, base_name) + '.conf.sample'
        if not os.path.isdir(os.path.dirname(os.path.abspath(out_file))):
            os.mkdir(os.path.dirname(os.path.abspath(out_file)))
    else:
        file_name = 'sample.config'
        out_file = os.path.join(app.srcdir, file_name)
    info('writing sample configuration to %s' % out_file)
    generator.main(args=['--config-file', config_path, '--output-file', out_file])