class PandasWrapper:

    @classmethod
    def create_dataframe(cls, data, columns):
        if not _with_pandas():
            raise Exception('DataFrames prototype requires pandas to function')
        return _pandas.DataFrame(data, columns=columns)

    @classmethod
    def is_dataframe(cls, data):
        if not _with_pandas():
            return False
        return isinstance(data, _pandas.core.frame.DataFrame)

    @classmethod
    def is_column(cls, data):
        if not _with_pandas():
            return False
        return isinstance(data, _pandas.core.series.Series)

    @classmethod
    def iterate(cls, data):
        if not _with_pandas():
            raise Exception('DataFrames prototype requires pandas to function')
        yield from data.itertuples(index=False)

    @classmethod
    def concat(cls, buffer):
        if not _with_pandas():
            raise Exception('DataFrames prototype requires pandas to function')
        return _pandas.concat(buffer)

    @classmethod
    def get_item(cls, data, idx):
        if not _with_pandas():
            raise Exception('DataFrames prototype requires pandas to function')
        return data[idx:idx + 1]

    @classmethod
    def get_len(cls, df):
        if not _with_pandas():
            raise Exception('DataFrames prototype requires pandas to function')
        return len(df.index)

    @classmethod
    def get_columns(cls, df):
        if not _with_pandas():
            raise Exception('DataFrames prototype requires pandas to function')
        return list(df.columns.values.tolist())