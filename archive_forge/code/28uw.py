"""
**1.5 Configuration Manager (`config_manager.py`):**
- **Purpose:** Manages external configuration settings.
- **Functions:**
  - `load_config(config_path)`: Loads configuration settings from a JSON/XML file.
"""


class ConfigManager:
    def load_config(self, config_path):
        """Loads configuration settings from a JSON file."""
        with open(config_path, "r") as file:
            return json.load(file)
