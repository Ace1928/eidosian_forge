import os
import sys
import time
from rdkit import RDConfig
def RunScript(script, doLongTests, verbose):
    if sys.argv[-1] == '-f':
        os.environ['PYTHON_TEST_FAILFAST'] = '1'
    if len(sys.argv) >= 3 and sys.argv[1] == '--testDir':
        os.chdir(sys.argv[2])
    if len(sys.argv) >= 5 and sys.argv[3] == '--buildType':
        os.environ[BUILD_TYPE_ENVVAR] = sys.argv[4].upper()
    if sys.path[0] != '.':
        sys.path = ['.'] + sys.path
    script = script.split('.py')[0]
    mod = __import__(script)
    try:
        tests = mod.tests
    except AttributeError:
        return ([], 0)
    longTests = []
    if doLongTests:
        try:
            longTests = mod.longTests
        except AttributeError:
            pass
    failed = []
    for i, entry in enumerate(tests):
        try:
            exeName, args, extras = entry
        except ValueError:
            print('bad entry:', entry)
            sys.exit(-1)
        try:
            res = RunTest(exeName, args, extras)
        except Exception:
            import traceback
            traceback.print_exc()
            res = TEST_FAILED
        if res != TEST_PASSED:
            failed.append((exeName, args, extras))
            if os.environ.get('PYTHON_TEST_FAILFAST', '') == '1':
                sys.stderr.write('Exiting from %s\n' % str([exeName] + list(args)))
                return (failed, i + 1)
    for i, (exeName, args, extras) in enumerate(longTests):
        res = RunTest(exeName, args, extras)
        if res != TEST_PASSED:
            failed.append((exeName, args, extras))
            if os.environ.get('PYTHON_TEST_FAILFAST', '') == '1':
                sys.stderr.write('Exitng from %s\n' % str([exeName] + list(args)))
                return (failed, len(tests) + i + 1)
    nTests = len(tests) + len(longTests)
    del sys.modules[script]
    if verbose and failed:
        for exeName, args, extras in failed:
            print('!!! TEST FAILURE: ', exeName, args, extras, file=sys.stderr)
    return (failed, nTests)