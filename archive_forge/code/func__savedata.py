import numbers
from . import _cluster  # type: ignore
def _savedata(self, jobname, gid, aid, geneindex, expindex):
    """Save the clustered data (PRIVATE)."""
    if self.genename is None:
        genename = self.geneid
    else:
        genename = self.genename
    ngenes, nexps = np.shape(self.data)
    with open(jobname + '.cdt', 'w') as outputfile:
        if self.mask is not None:
            mask = self.mask
        else:
            mask = np.ones((ngenes, nexps), int)
        if self.gweight is not None:
            gweight = self.gweight
        else:
            gweight = np.ones(ngenes)
        if self.eweight is not None:
            eweight = self.eweight
        else:
            eweight = np.ones(nexps)
        if gid:
            outputfile.write('GID\t')
        outputfile.write(self.uniqid)
        outputfile.write('\tNAME\tGWEIGHT')
        for j in expindex:
            outputfile.write(f'\t{self.expid[j]}')
        outputfile.write('\n')
        if aid:
            outputfile.write('AID')
            if gid:
                outputfile.write('\t')
            outputfile.write('\t\t')
            for j in expindex:
                outputfile.write('\tARRY%dX' % j)
            outputfile.write('\n')
        outputfile.write('EWEIGHT')
        if gid:
            outputfile.write('\t')
        outputfile.write('\t\t')
        for j in expindex:
            outputfile.write(f'\t{eweight[j]:f}')
        outputfile.write('\n')
        for i in geneindex:
            if gid:
                outputfile.write('GENE%dX\t' % i)
            outputfile.write(f'{self.geneid[i]}\t{genename[i]}\t{gweight[i]:f}')
            for j in expindex:
                outputfile.write('\t')
                if mask[i, j]:
                    outputfile.write(str(self.data[i, j]))
            outputfile.write('\n')