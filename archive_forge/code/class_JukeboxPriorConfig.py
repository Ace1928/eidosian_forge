import os
from typing import List, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class JukeboxPriorConfig(PretrainedConfig):
    """
        This is the configuration class to store the configuration of a [`JukeboxPrior`]. It is used to instantiate a
        `JukeboxPrior` according to the specified arguments, defining the model architecture. Instantiating a
        configuration with the defaults will yield a similar configuration to that of the top level prior from the
        [openai/jukebox-1b-lyrics](https://huggingface.co/openai/jukebox
    -1b-lyrics) architecture.

        Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
        documentation from [`PretrainedConfig`] for more information.



    Args:
        act_fn (`str`, *optional*, defaults to `"quick_gelu"`):
            Activation function.
        alignment_head (`int`, *optional*, defaults to 2):
            Head that is responsible of the alignment between lyrics and music. Only used to compute the lyric to audio
            alignment
        alignment_layer (`int`, *optional*, defaults to 68):
            Index of the layer that is responsible of the alignment between lyrics and music. Only used to compute the
            lyric to audio alignment
        attention_multiplier (`float`, *optional*, defaults to 0.25):
            Multiplier coefficient used to define the hidden dimension of the attention layers. 0.25 means that
            0.25*width of the model will be used.
        attention_pattern (`str`, *optional*, defaults to `"enc_dec_with_lyrics"`):
            Which attention pattern to use for the decoder/
        attn_dropout (`int`, *optional*, defaults to 0):
            Dropout probability for the post-attention layer dropout in the decoder.
        attn_res_scale (`bool`, *optional*, defaults to `False`):
            Whether or not to scale the residuals in the attention conditioner block.
        blocks (`int`, *optional*, defaults to 64):
            Number of blocks used in the `block_attn`. A sequence of length seq_len is factored as `[blocks, seq_len //
            blocks]` in the `JukeboxAttention` layer.
        conv_res_scale (`int`, *optional*):
            Whether or not to scale the residuals in the conditioner block. Since the top level prior does not have a
            conditioner, the default value is to None and should not be modified.
        num_layers (`int`, *optional*, defaults to 72):
            Number of layers of the transformer architecture.
        emb_dropout (`int`, *optional*, defaults to 0):
            Embedding dropout used in the lyric decoder.
        encoder_config (`JukeboxPriorConfig`, *optional*) :
            Configuration of the encoder which models the prior on the lyrics.
        encoder_loss_fraction (`float`, *optional*, defaults to 0.4):
            Multiplication factor used in front of the lyric encoder loss.
        hidden_size (`int`, *optional*, defaults to 2048):
            Hidden dimension of the attention layers.
        init_scale (`float`, *optional*, defaults to 0.2):
            Initialization scales for the prior modules.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether or not the prior is an encoder-decoder model. In case it is not, and `nb_relevant_lyric_tokens` is
            greater than 0, the `encoder` args should be specified for the lyric encoding.
        mask (`bool`, *optional*, defaults to `False`):
            Whether or not to mask the previous positions in the attention.
        max_duration (`int`, *optional*, defaults to 600):
            Maximum supported duration of the generated song in seconds.
        max_nb_genres (`int`, *optional*, defaults to 1):
            Maximum number of genres that can be used to condition the model.
        merged_decoder (`bool`, *optional*, defaults to `True`):
            Whether or not the decoder and the encoder inputs are merged. This is used for the separated
            encoder-decoder architecture
        metadata_conditioning (`bool`, *optional*, defaults to `True)`:
            Whether or not to condition on the artist and genre metadata.
        metadata_dims (`List[int]`, *optional*, defaults to `[604, 7898]`):
            Number of genres and the number of artists that were used to train the embedding layers of the prior
            models.
        min_duration (`int`, *optional*, defaults to 0):
            Minimum duration of the generated audio on which the model was trained.
        mlp_multiplier (`float`, *optional*, defaults to 1.0):
            Multiplier coefficient used to define the hidden dimension of the MLP layers. 0.25 means that 0.25*width of
            the model will be used.
        music_vocab_size (`int`, *optional*, defaults to 2048):
            Number of different music tokens. Should be similar to the `JukeboxVQVAEConfig.nb_discrete_codes`.
        n_ctx (`int`, *optional*, defaults to 6144):
            Number of context tokens for each prior. The context tokens are the music tokens that are attended to when
            generating music tokens.
        n_heads (`int`, *optional*, defaults to 2):
                Number of attention heads.
        nb_relevant_lyric_tokens (`int`, *optional*, defaults to 384):
            Number of lyric tokens that are used when sampling a single window of length `n_ctx`
        res_conv_depth (`int`, *optional*, defaults to 3):
            Depth of the `JukeboxDecoderConvBock` used to upsample the previously sampled audio in the
            `JukeboxMusicTokenConditioner`.
        res_conv_width (`int`, *optional*, defaults to 128):
            Width of the `JukeboxDecoderConvBock` used to upsample the previously sampled audio in the
            `JukeboxMusicTokenConditioner`.
        res_convolution_multiplier (`int`, *optional*, defaults to 1):
            Multiplier used to scale the `hidden_dim` of the `JukeboxResConv1DBlock`.
        res_dilation_cycle (`int`, *optional*):
            Dilation cycle used to define the `JukeboxMusicTokenConditioner`. Usually similar to the ones used in the
            corresponding level of the VQVAE. The first prior does not use it as it is not conditioned on upper level
            tokens.
        res_dilation_growth_rate (`int`, *optional*, defaults to 1):
            Dilation grow rate used between each convolutionnal block of the `JukeboxMusicTokenConditioner`
        res_downs_t (`List[int]`, *optional*, defaults to `[3, 2, 2]`):
            Downsampling rates used in the audio conditioning network
        res_strides_t (`List[int]`, *optional*, defaults to `[2, 2, 2]`):
            Striding used in the audio conditioning network
        resid_dropout (`int`, *optional*, defaults to 0):
            Residual dropout used in the attention pattern.
        sampling_rate (`int`, *optional*, defaults to 44100):
            Sampling rate used for training.
        spread (`int`, *optional*):
            Spread used in the `summary_spread_attention` pattern
        timing_dims (`int`, *optional*, defaults to 64):
            Dimension of the timing embedding.
        zero_out (`bool`, *optional*, defaults to `False`):
            Whether or not to zero out convolution weights when initializing.
    """
    model_type = 'jukebox_prior'
    attribute_map = {'max_position_embeddings': 'n_positions', 'num_attention_heads': 'n_head'}

    def __init__(self, act_fn='quick_gelu', level=0, alignment_head=2, alignment_layer=68, attention_multiplier=0.25, attention_pattern='enc_dec_with_lyrics', attn_dropout=0, attn_res_scale=False, blocks=64, conv_res_scale=None, num_layers=72, emb_dropout=0, encoder_config=None, encoder_loss_fraction=0.4, hidden_size=2048, init_scale=0.2, is_encoder_decoder=True, lyric_vocab_size=80, mask=False, max_duration=600, max_nb_genres=1, merged_decoder=True, metadata_conditioning=True, metadata_dims=[604, 7898], min_duration=0, mlp_multiplier=1.0, music_vocab_size=2048, n_ctx=6144, n_heads=2, nb_relevant_lyric_tokens=384, res_conv_depth=3, res_conv_width=128, res_convolution_multiplier=1, res_dilation_cycle=None, res_dilation_growth_rate=1, res_downs_t=[3, 2, 2], res_strides_t=[2, 2, 2], resid_dropout=0, sampling_rate=44100, spread=None, timing_dims=64, zero_out=False, **kwargs):
        self.act_fn = act_fn
        self.alignment_head = alignment_head
        self.alignment_layer = alignment_layer
        self.attention_multiplier = attention_multiplier
        self.attention_pattern = attention_pattern
        self.attn_dropout = attn_dropout
        self.attn_res_scale = attn_res_scale
        self.blocks = blocks
        self.conv_res_scale = conv_res_scale
        self.num_layers = num_layers
        self.emb_dropout = emb_dropout
        self.music_vocab_size = music_vocab_size
        if encoder_config is not None:
            self.encoder_config = JukeboxPriorConfig(**encoder_config)
        else:
            self.encoder_config = None
        self.encoder_loss_fraction = encoder_loss_fraction
        self.init_scale = init_scale
        self.is_encoder_decoder = is_encoder_decoder
        self.lyric_vocab_size = lyric_vocab_size
        self.level = level
        self.mask = mask
        self.max_duration = max_duration
        self.max_nb_genres = max_nb_genres
        self.merged_decoder = merged_decoder
        self.metadata_conditioning = metadata_conditioning
        self.metadata_dims = metadata_dims
        self.min_duration = min_duration
        self.mlp_multiplier = mlp_multiplier
        self.n_ctx = n_ctx
        self.n_heads = n_heads
        self.nb_relevant_lyric_tokens = nb_relevant_lyric_tokens
        self.res_conv_depth = res_conv_depth
        self.res_conv_width = res_conv_width
        self.res_convolution_multiplier = res_convolution_multiplier
        self.res_dilation_cycle = res_dilation_cycle
        self.res_dilation_growth_rate = res_dilation_growth_rate
        self.res_downs_t = res_downs_t
        self.res_strides_t = res_strides_t
        self.resid_dropout = resid_dropout
        self.sampling_rate = sampling_rate
        self.spread = spread
        self.timing_dims = timing_dims
        self.hidden_size = hidden_size
        self.zero_out = zero_out

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], level=0, **kwargs) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'jukebox':
            config_dict = config_dict[f'prior_{level}']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f'You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.')
        return cls.from_dict(config_dict, **kwargs)