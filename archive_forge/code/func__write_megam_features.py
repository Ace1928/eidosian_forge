import subprocess
from nltk.internals import find_binary
def _write_megam_features(vector, stream, bernoulli):
    if not vector:
        raise ValueError('MEGAM classifier requires the use of an always-on feature.')
    for fid, fval in vector:
        if bernoulli:
            if fval == 1:
                stream.write(' %s' % fid)
            elif fval != 0:
                raise ValueError('If bernoulli=True, then allfeatures must be binary.')
        else:
            stream.write(f' {fid} {fval}')