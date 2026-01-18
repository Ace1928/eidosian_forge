import torch
from typing import Any, Set, Dict, List, Tuple, OrderedDict
from collections import OrderedDict as OrdDict
def _get_plottable_data(self, feature_filter: str, module_fqn_filter: str) -> Tuple[List, List[List], bool]:
    """
        Takes in the feature filters and module filters and outputs the x and y data for plotting

        Args:
            feature_filter (str): Filters the features presented to only those that
                contain this filter substring
            module_fqn_filter (str): Only includes modules that contains this string

        Returns a tuple of three elements
            The first is a list containing relevant x-axis data
            The second is a list containing the corresponding y-axis data
            If the data is per channel
        """
    table_dict = self.generate_filtered_tables(feature_filter, module_fqn_filter)
    tensor_headers, tensor_table = table_dict[self.TABLE_TENSOR_KEY]
    channel_headers, channel_table = table_dict[self.TABLE_CHANNEL_KEY]
    tensor_info_features_count = len(tensor_headers) - ModelReportVisualizer.NUM_NON_FEATURE_TENSOR_HEADERS
    channel_info_features_count = len(channel_headers) - ModelReportVisualizer.NUM_NON_FEATURE_CHANNEL_HEADERS
    is_valid_per_tensor_plot: bool = tensor_info_features_count == 1
    is_valid_per_channel_plot: bool = channel_info_features_count == 1
    feature_column_offset = ModelReportVisualizer.NUM_NON_FEATURE_TENSOR_HEADERS
    table = tensor_table
    if is_valid_per_channel_plot:
        feature_column_offset = ModelReportVisualizer.NUM_NON_FEATURE_CHANNEL_HEADERS
        table = channel_table
    x_data: List = []
    y_data: List[List] = []
    if is_valid_per_tensor_plot:
        for table_row_num, row in enumerate(table):
            x_val_to_append = table_row_num
            tensor_feature_index = feature_column_offset
            row_value = row[tensor_feature_index]
            if not type(row_value) == str:
                x_data.append(x_val_to_append)
                y_data.append(row_value)
    elif is_valid_per_channel_plot:
        num_channels: int = max((row[self.CHANNEL_NUM_INDEX] for row in table)) + 1
        for channel in range(num_channels):
            y_data.append([])
        for table_row_num, row in enumerate(table):
            x_val_to_append = table_row_num
            current_channel = row[self.CHANNEL_NUM_INDEX]
            new_module_index: int = table_row_num // num_channels
            x_val_to_append = new_module_index
            tensor_feature_index = feature_column_offset
            row_value = row[tensor_feature_index]
            if not type(row_value) == str:
                if len(x_data) == 0 or x_data[-1] != x_val_to_append:
                    x_data.append(x_val_to_append)
                y_data[current_channel].append(row_value)
    else:
        error_str = 'Make sure to pick only a single feature with your filter to plot a graph.'
        error_str += ' We recommend calling get_all_unique_feature_names() to find unique feature names.'
        error_str += ' Pick one of those features to plot.'
        raise ValueError(error_str)
    return (x_data, y_data, is_valid_per_channel_plot)