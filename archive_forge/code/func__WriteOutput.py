import gyp.common
import json
import os
import posixpath
def _WriteOutput(params, **values):
    """Writes the output, either to stdout or a file is specified."""
    if 'error' in values:
        print('Error:', values['error'])
    if 'status' in values:
        print(values['status'])
    if 'targets' in values:
        values['targets'].sort()
        print('Supplied targets that depend on changed files:')
        for target in values['targets']:
            print('\t', target)
    if 'invalid_targets' in values:
        values['invalid_targets'].sort()
        print('The following targets were not found:')
        for target in values['invalid_targets']:
            print('\t', target)
    if 'build_targets' in values:
        values['build_targets'].sort()
        print('Targets that require a build:')
        for target in values['build_targets']:
            print('\t', target)
    if 'compile_targets' in values:
        values['compile_targets'].sort()
        print('Targets that need to be built:')
        for target in values['compile_targets']:
            print('\t', target)
    if 'test_targets' in values:
        values['test_targets'].sort()
        print('Test targets:')
        for target in values['test_targets']:
            print('\t', target)
    output_path = params.get('generator_flags', {}).get('analyzer_output_path', None)
    if not output_path:
        print(json.dumps(values))
        return
    try:
        f = open(output_path, 'w')
        f.write(json.dumps(values) + '\n')
        f.close()
    except OSError as e:
        print('Error writing to output file', output_path, str(e))