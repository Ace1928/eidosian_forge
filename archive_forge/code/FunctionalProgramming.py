
from functools import reduce

def apply_functional_programming_techniques(data):
    """
    Applies functional programming techniques to process data.

    Args:
    data (list): The data to be processed.

    Returns:
    tuple: A tuple containing the results of map, filter, and reduce operations.
    """

    # Using map with a lambda function to perform an operation on each data element
    mapped_data = list(map(lambda x: x * 2, data))

    # Using filter with a lambda function to filter the data
    filtered_data = list(filter(lambda x: x % 2 == 0, data))

    # Using reduce with a lambda function to reduce the data to a single value
    reduced_data = reduce(lambda x, y: x + y, data)

    return mapped_data, filtered_data, reduced_data

# Example usage
if __name__ == "__main__":
    sample_data = [1, 2, 3, 4, 5]
    results = apply_functional_programming_techniques(sample_data)
    print(f'Mapped Data: {results[0]}')
    print(f'Filtered Data: {results[1]}')
    print(f'Reduced Data: {results[2]}')
