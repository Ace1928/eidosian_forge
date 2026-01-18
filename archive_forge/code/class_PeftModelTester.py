import os
import tempfile
import unittest
import torch
from pytest import mark
from transformers import AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, is_peft_available
from .testing_utils import require_bitsandbytes, require_peft
@require_peft
@mark.peft_test
class PeftModelTester(unittest.TestCase):

    def setUp(self):
        self.causal_lm_model_id = 'trl-internal-testing/tiny-random-GPTNeoXForCausalLM'
        self.lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')

    def test_create_peft_model(self):
        """
        Simply creates a peft model and checks that it can be loaded.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)
        _ = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

    def test_peft_requires_grad(self):
        """
        Check that the value head of the returned model has requires_grad=True.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)
        assert model.v_head.summary.weight.requires_grad

    def test_check_peft_model_nb_trainable_params(self):
        """
        Check that the number of trainable parameters is correct.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)
        nb_trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
        assert nb_trainable_params == 10273
        non_peft_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.causal_lm_model_id)
        nb_trainable_params = sum((p.numel() for p in non_peft_model.parameters() if p.requires_grad))
        assert nb_trainable_params == 99578

    def test_create_peft_model_from_config(self):
        """
        Simply creates a peft model and checks that it can be loaded.
        """
        trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.causal_lm_model_id, peft_config=self.lora_config)
        nb_trainable_params = sum((p.numel() for p in trl_model.parameters() if p.requires_grad))
        assert nb_trainable_params == 10273
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(causal_lm_model, peft_config=self.lora_config)
        nb_trainable_params = sum((p.numel() for p in trl_model.parameters() if p.requires_grad))
        assert nb_trainable_params == 10273

    @require_bitsandbytes
    def test_create_bnb_peft_model_from_config(self):
        """
        Simply creates a peft model and checks that it can be loaded.
        """
        from bitsandbytes.nn import Linear8bitLt
        trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.causal_lm_model_id, peft_config=self.lora_config, load_in_8bit=True)
        nb_trainable_params = sum((p.numel() for p in trl_model.parameters() if p.requires_grad))
        assert nb_trainable_params == 10273
        assert trl_model.pretrained_model.model.gpt_neox.layers[0].mlp.dense_h_to_4h.__class__ == Linear8bitLt
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id, load_in_8bit=True, device_map='auto')
        trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(causal_lm_model, peft_config=self.lora_config)
        nb_trainable_params = sum((p.numel() for p in trl_model.parameters() if p.requires_grad))
        assert nb_trainable_params == 10273
        assert trl_model.pretrained_model.model.gpt_neox.layers[0].mlp.dense_h_to_4h.__class__ == Linear8bitLt

    def test_save_pretrained_peft(self):
        """
        Check that the model can be saved and loaded properly.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            assert os.path.isfile(f'{tmp_dir}/adapter_model.safetensors'), f'{tmp_dir}/adapter_model.safetensors does not exist'
            assert os.path.exists(f'{tmp_dir}/adapter_config.json'), f'{tmp_dir}/adapter_config.json does not exist'
            assert os.path.exists(f'{tmp_dir}/pytorch_model.bin'), f'{tmp_dir}/pytorch_model.bin does not exist'
            maybe_v_head = torch.load(f'{tmp_dir}/pytorch_model.bin')
            assert all((k.startswith('v_head') for k in maybe_v_head.keys())), f'keys in {tmp_dir}/pytorch_model.bin do not start with `v_head`'
            model_from_pretrained = AutoModelForCausalLMWithValueHead.from_pretrained(tmp_dir)
            for p1, p2 in zip(model.named_parameters(), model_from_pretrained.named_parameters()):
                assert torch.allclose(p1[1], p2[1]), f'{p1[0]} != {p2[0]}'

    def test_load_pretrained_peft(self):
        """
        Check that the model saved with peft class interface can be loaded properly.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)
        with tempfile.TemporaryDirectory() as tmp_dir:
            pretrained_model.save_pretrained(tmp_dir)
            model_from_pretrained = AutoModelForCausalLMWithValueHead.from_pretrained(tmp_dir)
            assert os.path.isfile(f'{tmp_dir}/adapter_model.safetensors'), f'{tmp_dir}/adapter_model.safetensors does not exist'
            assert os.path.exists(f'{tmp_dir}/adapter_config.json'), f'{tmp_dir}/adapter_config.json does not exist'
            for p1, p2 in zip(model.named_parameters(), model_from_pretrained.named_parameters()):
                if p1[0] not in ['v_head.summary.weight', 'v_head.summary.bias']:
                    assert torch.allclose(p1[1], p2[1]), f'{p1[0]} != {p2[0]}'

    def test_continue_training_peft_model(self):
        """
        Load peft and checks that it can continue training.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            pretrained_model.save_pretrained(tmp_dir)
            model = AutoModelForCausalLMWithValueHead.from_pretrained(tmp_dir, is_trainable=True)
            nb_trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
            assert nb_trainable_params == 10273