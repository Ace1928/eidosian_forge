from sklearn.datasets import load_breast_cancer
from ray import tune
from ray.data import read_datasource, Dataset, Datasource, ReadTask
from ray.data.block import BlockMetadata
from ray.tune.impl.utils import execute_dataset
class TestDatasource(Datasource):

    def prepare_read(self, parallelism: int, **read_args):
        import pyarrow as pa

        def load_data():
            data_raw = load_breast_cancer(as_frame=True)
            dataset_df = data_raw['data']
            dataset_df['target'] = data_raw['target']
            return [pa.Table.from_pandas(dataset_df)]
        meta = BlockMetadata(num_rows=None, size_bytes=None, schema=None, input_files=None, exec_stats=None)
        return [ReadTask(load_data, meta)]