import torch
from typing import Any, Set, Dict, List, Tuple, OrderedDict
from collections import OrderedDict as OrdDict
def _generate_channels_table(self, filtered_data: OrderedDict[str, Any], channel_features: List[str], num_channels: int) -> Tuple[List, List]:
    """
        Takes in the filtered data and features list and generates the channels headers and table

        Currently meant to generate the headers and table for both the channels information.

        Args:
            filtered_data (OrderedDict[str, Any]): An OrderedDict (sorted in order of model) mapping:
                module_fqns -> feature_names -> values
            channel_features (List[str]): A list of the channel level features
            num_channels (int): Number of channels in the channel data

        Returns a tuple with:
            A list of the headers of the channel table
            A list of lists containing the table information row by row
            The 0th index row will contain the headers of the columns
            The rest of the rows will contain data
        """
    channel_table: List[List[Any]] = []
    channel_headers: List[str] = []
    channel_table_entry_counter: int = 0
    if len(channel_features) > 0:
        for module_fqn in filtered_data:
            for channel in range(num_channels):
                new_channel_row = [channel_table_entry_counter, module_fqn, channel]
                for feature in channel_features:
                    if feature in filtered_data[module_fqn]:
                        feature_val = filtered_data[module_fqn][feature][channel]
                    else:
                        feature_val = 'Not Applicable'
                    if type(feature_val) is torch.Tensor:
                        feature_val = feature_val.item()
                    new_channel_row.append(feature_val)
                channel_table.append(new_channel_row)
                channel_table_entry_counter += 1
    if len(channel_table) != 0:
        channel_headers = ['idx', 'layer_fqn', 'channel'] + channel_features
    return (channel_headers, channel_table)