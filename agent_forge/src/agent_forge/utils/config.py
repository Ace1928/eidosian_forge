"""
Configuration management for Eidosian Forge.

Handles loading, validation, and management of system configuration.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import yaml

from ..models import ModelConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration for the Eidosian Forge system."""

    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "default.yaml",
    )

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, use default.
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = self._load_config()
        self._validate_config()
        logger.info(f"Configuration loaded from {self.config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.

        Returns:
            Dictionary containing configuration.
        """
        try:
            with open(self.config_path, "r") as f:
                if self.config_path.endswith(".yaml") or self.config_path.endswith(
                    ".yml"
                ):
                    return yaml.safe_load(f)
                elif self.config_path.endswith(".json"):
                    return json.load(f)
                else:
                    raise ValueError(
                        f"Unsupported config file format: {self.config_path}"
                    )
        except FileNotFoundError:
            logger.warning(
                f"Config file not found at {self.config_path}, creating default"
            )
            self._create_default_config()
            return self._load_config()

    def _create_default_config(self) -> None:
        """Create default configuration file if it doesn't exist."""
        default_config = {
            "agent": {
                "name": "Eidosian Forge",
                "version": "0.1.0",
                "description": "Recursive self-improving AI system",
            },
            "model": {
                "model_name": "Qwen/Qwen1.5-1.8B",
                "model_type": "qwen",
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "context_window": 8192,
                "parameters": {},
            },
            "memory": {
                "git_enabled": True,
                "git_repo_path": "./memory_repo",
                "commit_interval_minutes": 30,
                "importance_threshold": 0.5,
            },
            "execution": {
                "sandbox_enabled": True,
                "internet_access": True,
                "timeout_seconds": 60,
                "max_memory_mb": 512,
            },
            "logging": {"level": "INFO", "file": "eidosian_forge.log", "console": True},
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        # Write default config
        with open(self.config_path, "w") as f:
            if self.config_path.endswith(".yaml") or self.config_path.endswith(".yml"):
                yaml.dump(default_config, f, default_flow_style=False)
            elif self.config_path.endswith(".json"):
                json.dump(default_config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {self.config_path}")

        logger.info(f"Created default configuration at {self.config_path}")

    def _validate_config(self) -> None:
        """Validate configuration structure and values."""
        required_sections = ["agent", "model", "memory", "execution", "logging"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Additional validation can be added here

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Dot-separated configuration key (e.g., "model.temperature")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        parts = key.split(".")
        value = self.config

        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

    def get_model_config(self) -> ModelConfig:
        """
        Get model configuration.

        Returns:
            ModelConfig object with current configuration
        """
        model_config = self.config.get("model", {})
        return ModelConfig(
            model_name=model_config.get("model_name", "Qwen/Qwen1.5-1.8B"),
            model_type=model_config.get("model_type", "qwen"),
            max_tokens=model_config.get("max_tokens", 2048),
            temperature=model_config.get("temperature", 0.7),
            top_p=model_config.get("top_p", 0.9),
            context_window=model_config.get("context_window", 8192),
            parameters=model_config.get("parameters", {}),
        )

    def save(self) -> None:
        """Save current configuration to file."""
        with open(self.config_path, "w") as f:
            if self.config_path.endswith(".yaml") or self.config_path.endswith(".yml"):
                yaml.dump(self.config, f, default_flow_style=False)
            elif self.config_path.endswith(".json"):
                json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {self.config_path}")

        logger.info(f"Configuration saved to {self.config_path}")

    def update(self, key: str, value: Any) -> None:
        """
        Update configuration value.

        Args:
            key: Dot-separated configuration key (e.g., "model.temperature")
            value: New value
        """
        parts = key.split(".")
        config = self.config

        # Navigate to the deepest dict
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]

        # Update the value
        config[parts[-1]] = value
        logger.info(f"Updated configuration {key} = {value}")
