import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
class ReferenceModelTest(unittest.TestCase):

    def setUp(self):
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained('trl-internal-testing/tiny-random-GPT2LMHeadModel')
        self.test_input = torch.tensor([[0, 1, 2, 3]])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1)
        self.layer_format = 'pretrained_model.transformer.h.{layer}.attn.c_attn.weight'

    def test_independent_reference(self):
        layer_0 = self.layer_format.format(layer=0)
        layer_5 = self.layer_format.format(layer=4)
        ref_model = create_reference_model(self.model)
        first_layer_before = self.model.get_parameter(layer_0).data.clone()
        last_layer_before = self.model.get_parameter(layer_5).data.clone()
        first_ref_layer_before = ref_model.get_parameter(layer_0).data.clone()
        last_ref_layer_before = ref_model.get_parameter(layer_5).data.clone()
        output = self.model(input_ids=self.test_input, labels=self.test_input)
        output[1].backward()
        self.optimizer.step()
        first_layer_after = self.model.get_parameter(layer_0).data.clone()
        last_layer_after = self.model.get_parameter(layer_5).data.clone()
        first_ref_layer_after = ref_model.get_parameter(layer_0).data.clone()
        last_ref_layer_after = ref_model.get_parameter(layer_5).data.clone()
        assert (first_layer_before == first_ref_layer_before).all()
        assert (last_layer_before == last_ref_layer_before).all()
        assert (first_ref_layer_before == first_ref_layer_after).all()
        assert (last_ref_layer_before == last_ref_layer_after).all()
        assert not (first_layer_before == first_layer_after).all()
        assert not (last_layer_before == last_layer_after).all()

    def test_shared_layers(self):
        layer_0 = self.layer_format.format(layer=0)
        layer_1 = self.layer_format.format(layer=1)
        ref_model = create_reference_model(self.model, num_shared_layers=1)
        first_layer_before = self.model.get_parameter(layer_0).data.clone()
        second_layer_before = self.model.get_parameter(layer_1).data.clone()
        first_ref_layer_before = ref_model.get_parameter(layer_0).data.clone()
        second_ref_layer_before = ref_model.get_parameter(layer_1).data.clone()
        output = self.model(input_ids=self.test_input, labels=self.test_input)
        output[1].backward()
        self.optimizer.step()
        first_layer_after = self.model.get_parameter(layer_0).data.clone()
        second_layer_after = self.model.get_parameter(layer_1).data.clone()
        first_ref_layer_after = ref_model.get_parameter(layer_0).data.clone()
        second_ref_layer_after = ref_model.get_parameter(layer_1).data.clone()
        assert (first_layer_before == first_ref_layer_before).all()
        assert (second_layer_before == second_ref_layer_before).all()
        assert (first_ref_layer_before == first_ref_layer_after).all()
        assert (second_ref_layer_before == second_ref_layer_after).all()
        assert (first_layer_before == first_layer_after).all()
        assert not (second_layer_before == second_layer_after).all()