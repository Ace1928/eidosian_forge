from __future__ import annotations
import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Set, Tuple, TYPE_CHECKING, Union
from torch._inductor import config
from torch._inductor.utils import get_benchmark_name
@dataclass
class MetricTable:
    table_name: str
    column_names: List[str]
    num_rows_added: int = 0

    def add_row(self, row_fn):
        if self.table_name not in enabled_metric_tables():
            return
        row_dict = row_fn()
        assert len(self.column_names) == len(row_dict), f'{len(self.column_names)} v.s. {len(row_dict)}'
        assert set(self.column_names) == set(row_dict.keys()), f'{set(self.column_names)} v.s. {set(row_dict.keys())}'
        row = [get_benchmark_name()]
        row += [row_dict[column_name] for column_name in self.column_names]
        self._write_row(row)

    def output_filename(self):
        return f'metric_table_{self.table_name}.csv'

    def write_header(self):
        filename = self.output_filename()
        with open(filename, 'w') as fd:
            writer = csv.writer(fd, lineterminator='\n')
            writer.writerow(['model_name'] + self.column_names)

    def _write_row(self, row):
        filename = self.output_filename()
        if self.num_rows_added == 0 and (not os.path.exists(filename)):
            self.write_header()
        self.num_rows_added += 1
        for idx, orig_val in enumerate(row):
            if isinstance(orig_val, float):
                new_val = f'{orig_val:.6f}'
            elif orig_val is None:
                new_val = ''
            else:
                new_val = orig_val
            row[idx] = new_val
        with open(filename, 'a') as fd:
            writer = csv.writer(fd, lineterminator='\n')
            writer.writerow(row)

    @staticmethod
    def register_table(name, column_names):
        table = MetricTable(name, column_names)
        REGISTERED_METRIC_TABLES[name] = table