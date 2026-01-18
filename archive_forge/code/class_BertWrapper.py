from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.utils.fp16 import fp16_optimizer_wrapper
from parlai.utils.torch import neginf
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_
class BertWrapper(torch.nn.Module):
    """
    Adds a optional transformer layer and a linear layer on top of BERT.
    """

    def __init__(self, bert_model, output_dim, add_transformer_layer=False, layer_pulled=-1, aggregation='first'):
        super(BertWrapper, self).__init__()
        self.layer_pulled = layer_pulled
        self.aggregation = aggregation
        self.add_transformer_layer = add_transformer_layer
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)
        if add_transformer_layer:
            config_for_one_layer = BertConfig(0, hidden_size=bert_output_dim, num_attention_heads=int(bert_output_dim / 64), intermediate_size=3072, hidden_act='gelu')
            self.additional_transformer_layer = BertLayer(config_for_one_layer)
        self.additional_linear_layer = torch.nn.Linear(bert_output_dim, output_dim)
        self.bert_model = bert_model

    def forward(self, token_ids, segment_ids, attention_mask):
        """
        Forward pass.
        """
        output_bert, output_pooler = self.bert_model(token_ids, segment_ids, attention_mask)
        layer_of_interest = output_bert[self.layer_pulled]
        dtype = next(self.parameters()).dtype
        if self.add_transformer_layer:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (~extended_attention_mask).to(dtype) * neginf(dtype)
            embedding_layer = self.additional_transformer_layer(layer_of_interest, extended_attention_mask)
        else:
            embedding_layer = layer_of_interest
        if self.aggregation == 'mean':
            outputs_of_interest = embedding_layer[:, 1:, :]
            mask = attention_mask[:, 1:].type_as(embedding_layer).unsqueeze(2)
            sumed_embeddings = torch.sum(outputs_of_interest * mask, dim=1)
            nb_elems = torch.sum(attention_mask[:, 1:].type(dtype), dim=1).unsqueeze(1)
            embeddings = sumed_embeddings / nb_elems
        elif self.aggregation == 'max':
            outputs_of_interest = embedding_layer[:, 1:, :]
            mask = (~attention_mask[:, 1:]).type(dtype).unsqueeze(2) * neginf(dtype)
            embeddings, _ = torch.max(outputs_of_interest + mask, dim=1)
        else:
            embeddings = embedding_layer[:, 0, :]
        result = self.additional_linear_layer(embeddings)
        result += 0 * torch.sum(output_pooler)
        return result