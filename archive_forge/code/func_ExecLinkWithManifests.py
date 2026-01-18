import os
import re
import shutil
import subprocess
import stat
import string
import sys
def ExecLinkWithManifests(self, arch, embed_manifest, out, ldcmd, resname, mt, rc, intermediate_manifest, *manifests):
    """A wrapper for handling creating a manifest resource and then executing
    a link command."""
    variables = {'python': sys.executable, 'arch': arch, 'out': out, 'ldcmd': ldcmd, 'resname': resname, 'mt': mt, 'rc': rc, 'intermediate_manifest': intermediate_manifest, 'manifests': ' '.join(manifests)}
    add_to_ld = ''
    if manifests:
        subprocess.check_call('%(python)s gyp-win-tool manifest-wrapper %(arch)s %(mt)s -nologo -manifest %(manifests)s -out:%(out)s.manifest' % variables)
        if embed_manifest == 'True':
            subprocess.check_call('%(python)s gyp-win-tool manifest-to-rc %(arch)s %(out)s.manifest %(out)s.manifest.rc %(resname)s' % variables)
            subprocess.check_call('%(python)s gyp-win-tool rc-wrapper %(arch)s %(rc)s %(out)s.manifest.rc' % variables)
            add_to_ld = ' %(out)s.manifest.res' % variables
    subprocess.check_call(ldcmd + add_to_ld)
    if manifests:
        subprocess.check_call('%(python)s gyp-win-tool manifest-wrapper %(arch)s %(mt)s -nologo -manifest %(out)s.manifest %(intermediate_manifest)s -out:%(out)s.assert.manifest' % variables)
        assert_manifest = '%(out)s.assert.manifest' % variables
        our_manifest = '%(out)s.manifest' % variables
        with open(our_manifest) as our_f:
            with open(assert_manifest) as assert_f:
                translator = str.maketrans('', '', string.whitespace)
                our_data = our_f.read().translate(translator)
                assert_data = assert_f.read().translate(translator)
        if our_data != assert_data:
            os.unlink(out)

            def dump(filename):
                print(filename, file=sys.stderr)
                print('-----', file=sys.stderr)
                with open(filename) as f:
                    print(f.read(), file=sys.stderr)
                    print('-----', file=sys.stderr)
            dump(intermediate_manifest)
            dump(our_manifest)
            dump(assert_manifest)
            sys.stderr.write('Linker generated manifest "%s" added to final manifest "%s" (result in "%s"). Were /MANIFEST switches used in #pragma statements? ' % (intermediate_manifest, our_manifest, assert_manifest))
            return 1