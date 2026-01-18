from mlflow.protos.facet_feature_statistics_pb2 import Histogram
def _generate_equal_width_histogram_internal(bucket_left: float, bucket_right: float, quantiles, total_freq: float) -> Histogram.Bucket:
    """
    Generates a histogram bucket given the bucket range, the quantiles and the total frequency.

    Args:
        bucket_left: Bucket left boundary.
        bucket_right: Bucket right boundary.
        quantiles: The quantiles that capture the frequency distribution.
        total_freq: The total frequency (=count of rows).

    Returns:
        The histogram bucket corresponding to the inputs.
    """
    max_val = quantiles[-1]
    bucket_freq = 0.0
    quantile_freq = total_freq / (len(quantiles) - 1)
    for i in range(len(quantiles) - 1):
        quantile_low = quantiles[i]
        quantile_high = quantiles[i + 1]
        overlap_low = max(quantile_low, bucket_left)
        overlap_high = min(quantile_high, bucket_right)
        overlap_length = overlap_high - overlap_low
        quantile_contribution_ratio = 0.0
        if quantile_low == quantile_high:
            if bucket_left <= quantile_low < bucket_right or quantile_low == bucket_right == max_val:
                quantile_contribution_ratio = 1.0
        elif overlap_length > 0:
            quantile_contribution_ratio = overlap_length / (quantile_high - quantile_low)
        bucket_freq += quantile_freq * quantile_contribution_ratio
    return Histogram.Bucket(low_value=bucket_left, high_value=bucket_right, sample_count=bucket_freq)