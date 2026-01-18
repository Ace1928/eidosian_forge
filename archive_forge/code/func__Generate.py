import os
import subprocess
import sys
def _Generate(samples):
    insert_python_dir = os.getcwd()
    python_path = os.environ.get('PYTHONPATH')
    if python_path:
        python_path = os.pathsep.join([insert_python_dir, python_path])
    else:
        python_path = insert_python_dir
    os.environ['PYTHONPATH'] = python_path
    for sample in samples:
        sample_dir, sample_doc = os.path.split(sample)
        sample_dir = 'samples/' + sample_dir
        name, ext = os.path.splitext(sample_doc)
        if ext != '.json':
            raise RuntimeError('Expected .json discovery doc [{0}]'.format(sample))
        api_name, api_version = name.split('_')
        args = ['python', 'apitools/gen/gen_client.py', '--infile', 'samples/' + sample, '--init-file', 'empty', '--outdir={0}'.format(os.path.join(sample_dir, name)), '--overwrite', '--root_package', 'samples.{0}_sample.{0}_{1}'.format(api_name, api_version), 'client']
        sys.stderr.write('Running: {}\n'.format(' '.join(args)))
        subprocess.check_call(args)