from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
Returns a matrix to warp linear scale spectrograms to the mel scale.

        Returns a weight matrix that can be used to re-weight a tensor
        containing `num_spectrogram_bins` linearly sampled frequency information
        from `[0, sampling_rate / 2]` into `num_mel_bins` frequency information
        from `[lower_edge_hertz, upper_edge_hertz]` on the mel scale.

        This function follows the [Hidden Markov Model Toolkit (HTK)](
        http://htk.eng.cam.ac.uk/) convention, defining the mel scale in
        terms of a frequency in hertz according to the following formula:

        ```mel(f) = 2595 * log10( 1 + f/700)```

        In the returned matrix, all the triangles (filterbanks) have a peak
        value of 1.0.

        For example, the returned matrix `A` can be used to right-multiply a
        spectrogram `S` of shape `[frames, num_spectrogram_bins]` of linear
        scale spectrum values (e.g. STFT magnitudes) to generate a
        "mel spectrogram" `M` of shape `[frames, num_mel_bins]`.

        ```
        # `S` has shape [frames, num_spectrogram_bins]
        # `M` has shape [frames, num_mel_bins]
        M = keras.ops.matmul(S, A)
        ```

        The matrix can be used with `keras.ops.tensordot` to convert an
        arbitrary rank `Tensor` of linear-scale spectral bins into the
        mel scale.

        ```
        # S has shape [..., num_spectrogram_bins].
        # M has shape [..., num_mel_bins].
        M = keras.ops.tensordot(S, A, 1)
        ```

        References:
        - [Mel scale (Wikipedia)](https://en.wikipedia.org/wiki/Mel_scale)

        Args:
            num_mel_bins: Python int. How many bands in the resulting
                mel spectrum.
            num_spectrogram_bins: An integer `Tensor`. How many bins there are
                in the source spectrogram data, which is understood to be
                `fft_size // 2 + 1`, i.e. the spectrogram only contains the
                nonredundant FFT bins.
            sampling_rate: An integer or float `Tensor`. Samples per second of
                the input signal used to create the spectrogram. Used to figure
                out the frequencies corresponding to each spectrogram bin,
                which dictates how they are mapped into the mel scale.
            lower_edge_hertz: Python float. Lower bound on the frequencies to be
                included in the mel spectrum. This corresponds to the lower
                edge of the lowest triangular band.
            upper_edge_hertz: Python float. The desired top edge of the highest
                frequency band.
            dtype: The `DType` of the result matrix. Must be a floating point
                type.

        Returns:
            A tensor of shape `[num_spectrogram_bins, num_mel_bins]`.
        