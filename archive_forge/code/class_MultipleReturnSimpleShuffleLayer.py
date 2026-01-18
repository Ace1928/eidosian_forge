import operator
import warnings
import dask
from dask import core
from dask.core import istask
from dask.dataframe.core import _concat
from dask.dataframe.optimize import optimize
from dask.dataframe.shuffle import shuffle_group
from dask.highlevelgraph import HighLevelGraph
from .scheduler import MultipleReturnFunc, multiple_return_get
class MultipleReturnSimpleShuffleLayer(SimpleShuffleLayer):

    @classmethod
    def clone(cls, layer: SimpleShuffleLayer):
        return cls(name=layer.name, column=layer.column, npartitions=layer.npartitions, npartitions_input=layer.npartitions_input, ignore_index=layer.ignore_index, name_input=layer.name_input, meta_input=layer.meta_input, parts_out=layer.parts_out, annotations=layer.annotations)

    def __repr__(self):
        return f"MultipleReturnSimpleShuffleLayer<name='{self.name}', npartitions={self.npartitions}>"

    def __reduce__(self):
        attrs = ['name', 'column', 'npartitions', 'npartitions_input', 'ignore_index', 'name_input', 'meta_input', 'parts_out', 'annotations']
        return (MultipleReturnSimpleShuffleLayer, tuple((getattr(self, attr) for attr in attrs)))

    def _cull(self, parts_out):
        return MultipleReturnSimpleShuffleLayer(self.name, self.column, self.npartitions, self.npartitions_input, self.ignore_index, self.name_input, self.meta_input, parts_out=parts_out)

    def _construct_graph(self):
        """Construct graph for a simple shuffle operation."""
        shuffle_group_name = 'group-' + self.name
        shuffle_split_name = 'split-' + self.name
        dsk = {}
        n_parts_out = len(self.parts_out)
        for part_out in self.parts_out:
            _concat_list = [(shuffle_split_name, part_out, part_in) for part_in in range(self.npartitions_input)]
            dsk[self.name, part_out] = (_concat, _concat_list, self.ignore_index)
            for _, _part_out, _part_in in _concat_list:
                dsk[shuffle_split_name, _part_out, _part_in] = (multiple_return_get, (shuffle_group_name, _part_in), _part_out)
                if (shuffle_group_name, _part_in) not in dsk:
                    dsk[shuffle_group_name, _part_in] = (MultipleReturnFunc(shuffle_group, n_parts_out), (self.name_input, _part_in), self.column, 0, self.npartitions, self.npartitions, self.ignore_index, self.npartitions)
        return dsk