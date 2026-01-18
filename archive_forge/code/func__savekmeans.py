import numbers
from . import _cluster  # type: ignore
def _savekmeans(self, filename, clusterids, order, transpose):
    """Save the k-means clustering solution (PRIVATE)."""
    if transpose:
        label = 'ARRAY'
        names = self.expid
    else:
        label = self.uniqid
        names = self.geneid
    with open(filename, 'w') as outputfile:
        outputfile.write(label + '\tGROUP\n')
        index = np.argsort(order)
        n = len(names)
        sortedindex = np.zeros(n, int)
        counter = 0
        cluster = 0
        while counter < n:
            for j in index:
                if clusterids[j] == cluster:
                    outputfile.write(f'{names[j]}\t{cluster}\n')
                    sortedindex[counter] = j
                    counter += 1
            cluster += 1
    return sortedindex